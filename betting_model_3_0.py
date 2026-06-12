"""
=============================================================================
UNIVERSAL BETTING ENGINE v3.0 — BANEBET PRO
=============================================================================
Pełna integracja wszystkich 5 tierów zaawansowanych algorytmów.

NOWE W v3.0 vs v2.0:
  TIER 1: LightGBM + CatBoost + Kalibracja Isotonic/Platt (9 modeli ensemble)
  TIER 2: Stacking meta-learner (Ridge) zamiast ważonej MAE
  TIER 3: Optuna HPO + SHAP (pole why_bet w odpowiedzi)
  TIER 4: Dynamiczne Elo jako feature + Monte Carlo Correct Score
  TIER 5: Dixon-Coles fusion dla rynków OU* i CS_*

INSTALACJA:
  pip install numpy scipy scikit-learn xgboost lightgbm catboost
  pip install ngboost gpboost
  pip install optuna shap mapie
  pip install ktboost xfacpy  # opcjonalne
=============================================================================
"""

import numpy as np
import warnings
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from collections import deque
from scipy.special import expit
from scipy.special import gamma as scipy_gamma
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# IMPORTY OPCJONALNE — wszystkie z graceful fallback
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost niedostępny — fallback GradientBoosting.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("LightGBM niedostępny — pominięty w ensemble.")

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    warnings.warn("CatBoost niedostępny — pominięty w ensemble.")

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    NGB_AVAILABLE = True
except ImportError:
    NGB_AVAILABLE = False
    warnings.warn("NGBoost niedostępny.")

try:
    import gpboost as gpb
    GPB_AVAILABLE = True
except ImportError:
    GPB_AVAILABLE = False
    warnings.warn("GPBoost niedostępny.")

try:
    import KTBoost.KTBoost as KTBoost
    KTB_AVAILABLE = True
except ImportError:
    KTB_AVAILABLE = False

try:
    import xfacpy
    XFAC_AVAILABLE = True
except ImportError:
    XFAC_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna niedostępna — HPO pominięte.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP niedostępny — why_bet będzie używać feature importances.")


# =============================================================================
# BANEBET MATHEMATICAL LAYERS v1.0 (zachowane z v2.0)
# =============================================================================

def cubic_rational_transform(X: np.ndarray, B: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    return ((B + X + gamma) / (B + gamma)) ** 3

CubicRationalLayer = FunctionTransformer(func=cubic_rational_transform, kw_args={"B": 1.0, "gamma": 1.0}, validate=True)

def discriminant_switch(delta: float, p_base: float, B: float = 1.0, gamma: float = 1.0) -> Tuple[float, Optional[float]]:
    if delta > 0:
        discriminant_sqrt = np.sqrt(delta)
        sol_plus  = float(np.clip(p_base + discriminant_sqrt * gamma / (B + gamma), 0, 1))
        sol_minus = float(np.clip(p_base - discriminant_sqrt * gamma / (B + gamma), 0, 1))
        return sol_plus, sol_minus
    return float(np.clip(p_base, 0, 1)), None

NEGATIVE_GAMMA_TABLE: Dict[float, float] = {
    x: float(scipy_gamma(x)) for x in [-0.5, -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5]
}

def get_negative_gamma(x: float) -> float:
    if x in NEGATIVE_GAMMA_TABLE:
        return NEGATIVE_GAMMA_TABLE[x]
    try:
        return float(scipy_gamma(x))
    except Exception:
        return 1.0

def complex_gamma_weight(x: float, y: float) -> float:
    z = complex(x, y)
    try:
        return float(np.clip(abs(scipy_gamma(z)), 0.01, 10.0))
    except Exception:
        return 1.0

def negbinom_gamma_pmf(k: int, mu: float, gamma_neg: float = -0.5) -> float:
    from scipy.stats import poisson
    lam = max(mu, 0.01)
    base_pmf = poisson.pmf(k, lam)
    correction = abs(get_negative_gamma(gamma_neg))
    normalized_correction = np.clip(correction / (1.0 + correction), 0.5, 1.5)
    return float(np.clip(base_pmf * normalized_correction, 0.0, 1.0))

def identity_morphism(X: np.ndarray) -> np.ndarray:
    return X

IdentityMorphismLayer = FunctionTransformer(func=identity_morphism, validate=True)


# =============================================================================
# TIER 4: DYNAMICZNE ELO (feature wejściowa, nie model)
# =============================================================================

class DynamicElo:
    """
    Dynamiczne Elo K=32 jako feature wejściowa zamiast statycznego team_power.
    Przechowuje historię ratingów na podstawie wyników meczów.
    """
    def __init__(self, k: float = 32.0, base_rating: float = 1500.0):
        self.k = k
        self.base = base_rating
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expected_score(self, r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    def update(self, team_a: str, team_b: str, score_a: float) -> Tuple[float, float]:
        """
        score_a: 1=wygrana A, 0.5=remis, 0=przegrana A
        Zwraca (nowy_rating_A, nowy_rating_B)
        """
        r_a = self.get_rating(team_a)
        r_b = self.get_rating(team_b)
        e_a = self.expected_score(r_a, r_b)
        e_b = 1.0 - e_a

        self.ratings[team_a] = r_a + self.k * (score_a - e_a)
        self.ratings[team_b] = r_b + self.k * ((1 - score_a) - e_b)
        return self.ratings[team_a], self.ratings[team_b]

    def elo_feature(self, team_a: str, team_b: str) -> float:
        """
        Zwraca znormalizowany elo_diff w zakresie [0, 1] dla użycia jako feature.
        0.5 = mecz wyrównany, >0.5 = team_a silniejszy.
        """
        r_a = self.get_rating(team_a)
        r_b = self.get_rating(team_b)
        diff = r_a - r_b
        # Normalizacja: ±400 punktów elo → ±0.5 od środka
        return float(np.clip(0.5 + diff / 800.0, 0.0, 1.0))

    def to_dict(self) -> Dict:
        return dict(self.ratings)

    def from_dict(self, data: Dict):
        self.ratings = data


# =============================================================================
# TIER 5: DIXON-COLES FUSION dla rynków OU* i CS_*
# =============================================================================

class DixonColes:
    """
    Model Dixona-Colesa dla predykcji wyników piłkarskich.
    Estymuje mu_H (oczekiwana liczba goli drużyny domowej) i mu_A.
    Używany do generowania rozkładu prawdopodobieństw wyniku.
    """

    def __init__(self):
        self.mu_h: float = 1.5
        self.mu_a: float = 1.1
        self.rho: float = -0.1   # korekta korelacji dla 0-0, 1-0, 0-1, 1-1

    def fit_from_params(self, attack_h: float, defense_h: float,
                        attack_a: float, defense_a: float,
                        home_adv: float = 1.15) -> None:
        """
        Szybka parametryczna estymacja mu bez pełnego MLE.
        attack/defense: wartości w [0,1] z MEBN.
        """
        # Konwersja z przestrzeni [0,1] na Poisson lambda
        # attack=1 → ~2.5 goli, attack=0 → ~0.3 goli
        self.mu_h = float(np.clip(0.3 + attack_h * 2.2 * home_adv * (1.3 - defense_a), 0.3, 4.0))
        self.mu_a = float(np.clip(0.3 + attack_a * 2.2 * (1.3 - defense_h), 0.3, 4.0))

    def _tau(self, x: int, y: int) -> float:
        """Korekta Dixona-Colesa dla niskich wyników."""
        m, a = self.mu_h, self.mu_a
        r = self.rho
        if x == 0 and y == 0:
            return 1.0 - m * a * r
        if x == 1 and y == 0:
            return 1.0 + a * r
        if x == 0 and y == 1:
            return 1.0 + m * r
        if x == 1 and y == 1:
            return 1.0 - r
        return 1.0

    def score_probability(self, h: int, a: int) -> float:
        """P(home=h, away=a) według modelu Dixona-Colesa."""
        from scipy.stats import poisson
        p = poisson.pmf(h, self.mu_h) * poisson.pmf(a, self.mu_a) * self._tau(h, a)
        return float(max(0.0, p))

    def correct_score_distribution(self, max_goals: int = 5) -> Dict[str, float]:
        """Rozkład prawdopodobieństwa dla wszystkich wyników CS."""
        dist = {}
        total = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = self.score_probability(h, a)
                dist[f"CS_{h}{a}"] = p
                total += p
        # Normalizacja + CS_OTHER
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        dist["CS_OTHER"] = max(0.0, 1.0 - sum(
            dist.get(f"CS_{h}{a}", 0) for h in range(max_goals + 1) for a in range(max_goals + 1)
        ))
        return dist

    def over_under_probs(self) -> Dict[str, float]:
        """Prawdopodobieństwa dla rynków OU 0.5 do 5.5."""
        dist = self.correct_score_distribution()
        total_goals: Dict[int, float] = {}
        for key, p in dist.items():
            if key == "CS_OTHER":
                continue
            h = int(key[3])
            a = int(key[4])
            t = h + a
            total_goals[t] = total_goals.get(t, 0) + p

        ou = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            p_over = sum(v for k, v in total_goals.items() if k > line)
            ou[f"OU{str(line).replace('.', '')}"] = round(float(np.clip(p_over, 0, 1)), 4)
        return ou

    def btts_prob(self) -> float:
        """P(oba strzelają ≥1 gol)."""
        p_h0 = self.score_probability(0, 0) + sum(self.score_probability(0, a) for a in range(1, 6))
        p_a0 = sum(self.score_probability(h, 0) for h in range(1, 6))
        return float(np.clip(1.0 - (p_h0 + p_a0 - self.score_probability(0, 0)), 0, 1))


# =============================================================================
# TIER 4: MONTE CARLO CORRECT SCORE
# =============================================================================

def monte_carlo_correct_score(mu_h: float, mu_a: float,
                               n_sims: int = 10000,
                               max_goals: int = 5) -> Dict[str, float]:
    """
    Monte Carlo symulacja meczu → empiryczny rozkład CS.
    Szybsza i bardziej realistyczna niż analityczny Poisson przy dużych mu.
    """
    np.random.seed(None)  # różne nasiona każdy call
    goals_h = np.random.poisson(mu_h, n_sims)
    goals_a = np.random.poisson(mu_a, n_sims)

    dist: Dict[str, float] = {}
    total = n_sims

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            count = int(np.sum((goals_h == h) & (goals_a == a)))
            dist[f"CS_{h}{a}"] = round(count / total, 4)

    other_count = int(np.sum((goals_h > max_goals) | (goals_a > max_goals)))
    dist["CS_OTHER"] = round(other_count / total, 4)
    return dist


# =============================================================================
# TIER 1: KALIBRACJA PRAWDOPODOBIEŃSTW (Isotonic / Platt)
# =============================================================================

class ProbabilityCalibrator:
    """
    Kalibracja wyjść modeli ensemble — zamienia surowe p_win na skalibrowane.
    Isotonic Regression jest mocniejsza dla dużych zbiorów (>1000 próbek).
    Platt Scaling (logistic) dla małych.
    """
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None
        self.fitted = False

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> None:
        """
        p_raw: surowe predykcje [0,1] z ensemble
        y_true: rzeczywiste etykiety binarne {0, 1}
        """
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(p_raw, y_true)
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(C=1.0)
            self.calibrator.fit(p_raw.reshape(-1, 1), y_true)
        self.fitted = True

    def calibrate(self, p_raw: float) -> float:
        if not self.fitted:
            return p_raw
        if self.method == "isotonic":
            return float(np.clip(self.calibrator.predict([p_raw])[0], 0.01, 0.99))
        elif self.method == "platt":
            return float(self.calibrator.predict_proba([[p_raw]])[0, 1])
        return p_raw

    def calibrate_array(self, p_raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return p_raw
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(p_raw), 0.01, 0.99)
        elif self.method == "platt":
            return self.calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw


# =============================================================================
# TIER 3: OPTUNA HPO
# =============================================================================

def tune_xgb_params(X: np.ndarray, y: np.ndarray,
                     n_trials: int = 30) -> Dict:
    """
    Bayesowski HPO przez Optuna dla XGBoost.
    n_trials=30 → ~2 minuty, n_trials=100 → ~6 minut.
    """
    if not OPTUNA_AVAILABLE or not XGB_AVAILABLE:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        maes = []
        for tr, val in kf.split(X):
            model.fit(X[tr], y[tr])
            pred = np.clip(model.predict(X[val]), 0, 1)
            maes.append(np.mean(np.abs(pred - y[val])))
        return np.mean(maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[OPTUNA] Najlepsze params XGB: {study.best_params} | MAE={study.best_value:.4f}")
    return study.best_params


def tune_lgb_params(X: np.ndarray, y: np.ndarray,
                     n_trials: int = 30) -> Dict:
    """Bayesowski HPO dla LightGBM."""
    if not OPTUNA_AVAILABLE or not LGB_AVAILABLE:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "mae",
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        maes = []
        for tr, val in kf.split(X):
            model.fit(X[tr], y[tr])
            pred = np.clip(model.predict(X[val]), 0, 1)
            maes.append(np.mean(np.abs(pred - y[val])))
        return np.mean(maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[OPTUNA] Najlepsze params LGB: {study.best_params} | MAE={study.best_value:.4f}")
    return study.best_params


# =============================================================================
# KONFIGURACJA SPORTÓW (zachowana z v2.0, rozszerzona o attack/defense)
# =============================================================================

@dataclass
class SportConfig:
    name: str
    dimensions: List[str]
    weights: List[float]
    markets: List[str]
    has_sets: bool = False
    weather_relevant: bool = False
    # Indeksy wymiarów attack/defense dla Dixon-Coles (None = brak)
    attack_idx: Optional[int] = None
    defense_idx: Optional[int] = None


SPORT_CONFIGS: Dict[str, SportConfig] = {

    "football": SportConfig(
        name="Piłka Nożna",
        dimensions=["team_power", "home_advantage", "fatigue", "form",
                    "head2head", "attack", "defense", "pressure"],
        weights=[0.30, 0.15, 0.10, 0.20, 0.10, 0.05, 0.05, 0.05],
        markets=[
            "1X2", "DC_1X", "DC_X2", "DC_12",
            "BTTS_YES", "BTTS_NO",
            "OU05", "OU15", "OU25", "OU35", "OU45", "OU55",
            "AH05", "AH_M1", "AH_P1",
            "DNB_H", "DNB_A",
            "HT_1X2",
            "HTFT_11", "HTFT_1X", "HTFT_12",
            "HTFT_X1", "HTFT_XX", "HTFT_X2",
            "HTFT_21", "HTFT_2X", "HTFT_22",
            "CS_00", "CS_10", "CS_01", "CS_11",
            "CS_20", "CS_02", "CS_21", "CS_12",
            "CS_22", "CS_30", "CS_03", "CS_31", "CS_13",
            "CS_OTHER",
            "TG_H_0", "TG_H_1", "TG_H_2P",
            "TG_A_0", "TG_A_1", "TG_A_2P",
            "HT_OU05", "HT_OU15",
            "BTTS_WIN", "BTTS_DRAW", "BTTS_LOSE",
        ],
        weather_relevant=False,
        attack_idx=5,   # indeks wymiaru "attack"
        defense_idx=6,  # indeks wymiaru "defense"
    ),

    "basketball": SportConfig(
        name="Koszykówka",
        dimensions=["team_power", "home_advantage", "fatigue", "shooting_pct",
                    "pace", "injury_impact", "form", "back2back"],
        weights=[0.28, 0.12, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05],
        markets=["1X2", "Handicap", "Total_OU", "1H_OU", "Race_to_X", "Player_Props", "Q1_OU"],
    ),

    "hockey": SportConfig(
        name="Hokej na Lodzie",
        dimensions=["team_power", "home_advantage", "goalie_rating", "power_play",
                    "fatigue", "form", "shots_ratio", "penalty_kill"],
        weights=[0.25, 0.12, 0.20, 0.12, 0.08, 0.10, 0.08, 0.05],
        markets=["1X2", "Handicap_Puck", "Total_OU", "1P_OU",
                 "First_Goal_Method", "Correct_Score", "Shootout"],
    ),

    "tennis": SportConfig(
        name="Tenis",
        dimensions=["player_power", "surface_affinity", "fatigue", "rank_diff",
                    "form_last5", "head2head", "mental_strength", "serve_rating"],
        weights=[0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05],
        markets=["Match_Winner", "Set_Handicap", "Total_Sets", "Correct_Score_Sets",
                 "First_Set_Winner", "Games_OU", "Ace_Count"],
        has_sets=True,
    ),

    "volleyball": SportConfig(
        name="Siatkówka",
        dimensions=["team_power", "home_advantage", "fatigue", "block_rating",
                    "serve_power", "form", "head2head", "reception"],
        weights=[0.25, 0.12, 0.12, 0.12, 0.12, 0.10, 0.10, 0.07],
        markets=["1X2", "Set_Handicap", "Total_Sets", "Correct_Score_Sets",
                 "Set1_Winner", "Points_OU"],
        has_sets=True,
    ),

    "baseball": SportConfig(
        name="Baseball",
        dimensions=["team_power", "pitcher_rating", "bullpen_strength", "batting_avg",
                    "home_advantage", "fatigue", "park_factor", "weather"],
        weights=[0.20, 0.25, 0.12, 0.12, 0.08, 0.08, 0.08, 0.07],
        markets=["Moneyline", "Run_Line", "Total_OU", "1st_Inning_OU",
                 "F5_Winner", "NRFI", "Player_HR"],
        weather_relevant=True,
    ),

    "american_football": SportConfig(
        name="Futbol Amerykański",
        dimensions=["team_power", "home_advantage", "QB_rating", "defense_rank",
                    "fatigue", "form", "turnover_diff", "red_zone_pct"],
        weights=[0.22, 0.12, 0.22, 0.15, 0.08, 0.08, 0.08, 0.05],
        markets=["Moneyline", "Spread", "Total_OU", "1H_OU",
                 "First_TD", "Player_Props_Rushing", "Player_Props_Passing"],
        weather_relevant=True,
    ),

    "rugby": SportConfig(
        name="Rugby",
        dimensions=["team_power", "home_advantage", "scrum_strength", "lineout_win",
                    "fatigue", "form", "discipline", "weather"],
        weights=[0.25, 0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.08],
        markets=["1X2", "Handicap", "Total_OU", "HT_Winner",
                 "First_Try_Scorer", "Cards_OU", "Total_Tries"],
        weather_relevant=True,
    ),

    "cycling": SportConfig(
        name="Kolarstwo",
        dimensions=["rider_power", "climb_rating", "sprint_rating", "team_support",
                    "fatigue", "route_affinity", "current_form", "weather"],
        weights=[0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05],
        markets=["Stage_Winner", "Top3_Finish", "GC_Leader",
                 "Points_Leader", "KOM_Leader", "Team_Winner"],
        weather_relevant=True,
    ),

    "boxing_mma": SportConfig(
        name="Boks / MMA",
        dimensions=["fighter_power", "reach_advantage", "cardio", "striking_acc",
                    "grappling", "experience", "mental_strength", "camp_quality"],
        weights=[0.25, 0.10, 0.15, 0.15, 0.12, 0.10, 0.08, 0.05],
        markets=["Fight_Winner", "Method_of_Victory", "Round_Betting",
                 "Goes_Distance", "Round_OU", "Knockdown"],
    ),

    "snooker": SportConfig(
        name="Snooker",
        dimensions=["player_power", "form_last5", "ranking_diff", "safety_play",
                    "break_building", "mental_strength", "head2head", "fatigue"],
        weights=[0.25, 0.15, 0.12, 0.12, 0.12, 0.10, 0.09, 0.05],
        markets=["Match_Winner", "Frame_Handicap", "Total_Frames",
                 "First_Century", "Highest_Break", "Correct_Score_Frames"],
        has_sets=True,
    ),

    "darts": SportConfig(
        name="Rzutki (Darts)",
        dimensions=["player_power", "three_dart_avg", "checkout_pct",
                    "form_last5", "leg_win_ratio", "head2head",
                    "crowd_impact", "mental_strength"],
        weights=[0.20, 0.25, 0.18, 0.12, 0.10, 0.07, 0.05, 0.03],
        markets=["Match_Winner", "Leg_Handicap", "Total_Legs",
                 "180s_OU", "First_Leg", "Correct_Score_Legs"],
        has_sets=True,
    ),

    "esports": SportConfig(
        name="E-Sport",
        dimensions=["team_elo", "map_winrate", "agent_pool", "economy_rating",
                    "recent_patch_impact", "head2head", "meta_adaptation", "mental"],
        weights=[0.25, 0.18, 0.12, 0.12, 0.12, 0.08, 0.08, 0.05],
        markets=["Match_Winner", "Map_Handicap", "Total_Maps",
                 "First_Blood", "Pistol_Round", "Correct_Score_Maps"],
    ),

    "racing": SportConfig(
        name="Wyścigi (F1/Konie/Greyhound)",
        dimensions=["driver_power", "car_performance", "track_affinity",
                    "qualifying_pos", "pit_strategy", "weather",
                    "tire_compound", "reliability"],
        weights=[0.20, 0.22, 0.15, 0.12, 0.10, 0.08, 0.08, 0.05],
        markets=["Race_Winner", "Podium_Finish", "Top6", "DNF",
                 "Fastest_Lap", "Pole_Position", "H2H_Driver"],
        weather_relevant=True,
    ),

    "cricket": SportConfig(
        name="Krykiet",
        dimensions=["team_power", "batting_strength", "bowling_strength", "pitch_type",
                    "weather", "form", "head2head", "toss_impact"],
        weights=[0.20, 0.18, 0.18, 0.12, 0.10, 0.08, 0.08, 0.06],
        markets=["Match_Winner", "Total_Runs_OU", "Top_Batsman",
                 "Top_Bowler", "Innings_Runs", "Fall_of_Wicket"],
        weather_relevant=True,
    ),
}


# =============================================================================
# MEBN — Bayesian Entity Network (zachowany z v2.0)
# =============================================================================

class SportMEBN:
    def __init__(self, config: SportConfig, tree_weights: Optional[Tuple[float, float]] = None):
        self.config = config
        n = len(config.dimensions)
        self.w = np.array(config.weights[:n])
        self.w /= self.w.sum()
        self.xgb_w = tree_weights[0] if tree_weights else 0.5
        self.rf_w  = tree_weights[1] if tree_weights else 0.5
        _v0 = np.array([0.5] * n)
        _lin0 = float(np.dot(self.w, _v0))
        _syn0 = sum(_v0[i] * _v0[i+1] * self.w[i] * self.w[i+1] for i in range(n-1))
        _pt0  = float(np.sin(_v0[-1] * np.pi) * 0.05)
        self._neutral_offset = _lin0 + _syn0 * 0.5 + _pt0

    def probability_function(self, coords: np.ndarray) -> float:
        n = len(self.config.dimensions)
        coords = np.clip(coords[:n], 0.0, 1.0)
        linear = float(np.dot(self.w, coords))
        synergy = sum(coords[i] * coords[i+1] * self.w[i] * self.w[i+1] for i in range(n-1))
        pressure_term = np.sin(coords[-1] * np.pi) * 0.05
        total = linear + synergy * 0.5 + pressure_term
        return float(np.clip(expit((total - self._neutral_offset) * 5.0), 0.0, 1.0))


# =============================================================================
# TIER 1+2+3: TREE LAYER v3 — 9 modeli + Stacking + Optuna + SHAP
# =============================================================================

class TreeLayerV3:
    """
    Extreme Ensemble 9-modelowy:
      1. XGBoost Extreme  (+ Optuna HPO)
      2. LightGBM         (+ Optuna HPO)  ← NOWY
      3. CatBoost                          ← NOWY
      4. KTBoost
      5. NGBoost
      6. GPBoost
      7. RandomForest (residual corrector)
      + Stacking meta-learner (Ridge na OOF)  ← NOWY
      + Kalibracja Isotonic                   ← NOWA
    """

    def __init__(self, use_optuna: bool = True, optuna_trials: int = 30,
                 calibration_method: str = "isotonic"):
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.optuna_trials = optuna_trials
        self.calibrator = ProbabilityCalibrator(method=calibration_method)

        # XGBoost
        if XGB_AVAILABLE:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                objective="reg:absoluteerror", tree_method="hist", verbosity=0,
            )
        else:
            self.xgb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)

        # LightGBM
        if LGB_AVAILABLE:
            self.lgb_model = lgb.LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, objective="mae", verbose=-1,
            )
        else:
            self.lgb_model = None

        # CatBoost
        if CB_AVAILABLE:
            self.cb_model = cb.CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.03,
                loss_function="MAE", verbose=0, random_seed=42,
            )
        else:
            self.cb_model = None

        # KTBoost
        if KTB_AVAILABLE:
            self.ktb_model = KTBoost.BoostingRegressor(
                loss="mse", n_estimators=200, learning_rate=0.05,
                base_learner="kernel", kernel="rbf",
            )
        else:
            self.ktb_model = None

        # NGBoost
        if NGB_AVAILABLE:
            self.ngb_model = NGBRegressor(
                Dist=Normal, n_estimators=200, learning_rate=0.05,
                verbose=False, random_state=42,
            )
        else:
            self.ngb_model = None

        # GPBoost
        self.gpb_model = None
        if GPB_AVAILABLE:
            self._gpb_params = {
                "num_iterations": 200, "learning_rate": 0.05, "max_depth": 5,
                "min_data_in_leaf": 5, "objective": "regression_l1", "verbose": -1,
            }

        # RandomForest
        self.rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=8, bootstrap=True, n_jobs=-1, random_state=42,
        )

        # Stacking meta-learner (Ridge na predykcjach OOF)
        self.meta_learner = Ridge(alpha=1.0)

        self.trained = False
        self._shap_explainer = None
        self._feature_names: List[str] = []
        self._xgb_importance: float = 0.5
        self._rf_importance:  float = 0.5
        self._active: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray,
              n_splits: int = 5, feature_names: Optional[List[str]] = None) -> None:
        """
        K-fold OOF dla wszystkich modeli → stacking meta-learner.
        """
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # TIER 3: Optuna HPO przed treningiem
        if self.use_optuna and len(X) >= 100:
            print("[OPTUNA] Strojenie XGB...")
            best_xgb = tune_xgb_params(X, y, n_trials=self.optuna_trials)
            if best_xgb and XGB_AVAILABLE:
                best_xgb.update({"objective": "reg:absoluteerror", "tree_method": "hist", "verbosity": 0})
                self.xgb_model = xgb.XGBRegressor(**best_xgb)

            if LGB_AVAILABLE:
                print("[OPTUNA] Strojenie LGB...")
                best_lgb = tune_lgb_params(X, y, n_trials=self.optuna_trials)
                if best_lgb:
                    best_lgb.update({"objective": "mae", "verbose": -1})
                    self.lgb_model = lgb.LGBMRegressor(**best_lgb)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        n = len(y)

        # OOF arrays dla stacking
        oof_dict: Dict[str, np.ndarray] = {
            "xgb": np.zeros(n), "lgb": np.zeros(n), "cb":  np.zeros(n),
            "ktb": np.zeros(n), "ngb": np.zeros(n), "gpb": np.zeros(n),
        }
        self._active = {
            "xgb": True,
            "lgb": LGB_AVAILABLE and self.lgb_model is not None,
            "cb":  CB_AVAILABLE  and self.cb_model  is not None,
            "ktb": KTB_AVAILABLE and self.ktb_model is not None,
            "ngb": NGB_AVAILABLE and self.ngb_model is not None,
            "gpb": GPB_AVAILABLE,
        }

        print(f"[ENSEMBLE v3] Aktywne modele: " + " | ".join(f"{k}={v}" for k, v in self._active.items()))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            Xt, Xv = X[tr_idx], X[val_idx]
            yt = y[tr_idx]

            # XGBoost
            self.xgb_model.fit(Xt, yt)
            oof_dict["xgb"][val_idx] = np.clip(self.xgb_model.predict(Xv), 0, 1)

            # LightGBM
            if self._active["lgb"]:
                try:
                    self.lgb_model.fit(Xt, yt)
                    oof_dict["lgb"][val_idx] = np.clip(self.lgb_model.predict(Xv), 0, 1)
                except Exception as e:
                    warnings.warn(f"[LGB fold {fold}] {e}")
                    self._active["lgb"] = False

            # CatBoost
            if self._active["cb"]:
                try:
                    self.cb_model.fit(Xt, yt)
                    oof_dict["cb"][val_idx] = np.clip(self.cb_model.predict(Xv), 0, 1)
                except Exception as e:
                    warnings.warn(f"[CB fold {fold}] {e}")
                    self._active["cb"] = False

            # KTBoost
            if self._active["ktb"]:
                try:
                    self.ktb_model.fit(Xt, yt)
                    oof_dict["ktb"][val_idx] = np.clip(self.ktb_model.predict(Xv), 0, 1)
                except Exception as e:
                    warnings.warn(f"[KTB fold {fold}] {e}")
                    self._active["ktb"] = False

            # NGBoost
            if self._active["ngb"]:
                try:
                    self.ngb_model.fit(Xt, yt)
                    oof_dict["ngb"][val_idx] = np.clip(self.ngb_model.predict(Xv), 0, 1)
                except Exception as e:
                    warnings.warn(f"[NGB fold {fold}] {e}")
                    self._active["ngb"] = False

            # GPBoost
            if self._active["gpb"]:
                try:
                    gpb_data = gpb.Dataset(Xt, label=yt)
                    _gp = gpb.GPModel(num_data=len(yt), likelihood="gaussian")
                    _bst = gpb.train(params=self._gpb_params, train_set=gpb_data,
                                     gp_model=_gp, num_boost_round=self._gpb_params["num_iterations"])
                    gpb_pred = _bst.predict(data=Xv, gp_coords_pred=Xv, predict_var=False)
                    if isinstance(gpb_pred, dict):
                        gpb_pred = gpb_pred.get("response_mean", list(gpb_pred.values())[0])
                    oof_dict["gpb"][val_idx] = np.clip(gpb_pred, 0, 1)
                except Exception as e:
                    warnings.warn(f"[GPB fold {fold}] {e}")
                    self._active["gpb"] = False

        # --- TIER 2: Stacking meta-learner (Ridge na OOF) ---
        # Budujemy macierz OOF z aktywnych modeli
        oof_cols = [oof_dict[k] for k, v in self._active.items() if v]
        if len(oof_cols) >= 2:
            oof_matrix = np.column_stack(oof_cols)
            self.meta_learner.fit(oof_matrix, y)
            print(f"[STACKING] Meta-learner Ridge wytrenowany na {oof_matrix.shape[1]} modelach OOF.")
        else:
            self.meta_learner = None
            print("[STACKING] Za mało modeli dla meta-learnera — fallback do ważonej średniej.")

        # --- Trening finalny na całych danych ---
        self.xgb_model.fit(X, y)

        if self._active["lgb"]:
            try: self.lgb_model.fit(X, y)
            except Exception as e: warnings.warn(f"[LGB final] {e}"); self._active["lgb"] = False

        if self._active["cb"]:
            try: self.cb_model.fit(X, y)
            except Exception as e: warnings.warn(f"[CB final] {e}"); self._active["cb"] = False

        if self._active["ktb"]:
            try: self.ktb_model.fit(X, y)
            except Exception as e: warnings.warn(f"[KTB final] {e}"); self._active["ktb"] = False

        if self._active["ngb"]:
            try: self.ngb_model.fit(X, y)
            except Exception as e: warnings.warn(f"[NGB final] {e}"); self._active["ngb"] = False

        if self._active["gpb"]:
            try:
                gpb_data_full = gpb.Dataset(X, label=y)
                _gp_full = gpb.GPModel(num_data=len(y), likelihood="gaussian")
                self.gpb_model = gpb.train(
                    params=self._gpb_params, train_set=gpb_data_full,
                    gp_model=_gp_full, num_boost_round=self._gpb_params["num_iterations"],
                )
            except Exception as e:
                warnings.warn(f"[GPB final] {e}"); self._active["gpb"] = False

        # RF na residuach XGB
        xgb_preds_train = np.clip(self.xgb_model.predict(X), 0, 1)
        residuals = y - xgb_preds_train
        self.rf_model.fit(X, residuals)

        # Feature importances
        self._xgb_importance = float(self.xgb_model.feature_importances_.mean())
        self._rf_importance   = float(self.rf_model.feature_importances_.mean())

        # TIER 1: Kalibracja — używamy OOF predykcji XGB
        # Konwertujemy y na etykiety binarne (>0.5 = wygrana) do kalibracji
        y_binary = (y > 0.5).astype(float)
        self.calibrator.fit(oof_dict["xgb"], y_binary)
        print("[CALIBRATION] Kalibracja Isotonic wytrenowana na OOF XGB.")

        # TIER 3: SHAP explainer
        if SHAP_AVAILABLE:
            try:
                self._shap_explainer = shap.TreeExplainer(self.xgb_model)
                print("[SHAP] Explainer zainicjalizowany.")
            except Exception as e:
                warnings.warn(f"[SHAP] {e}")

        self.trained = True
        print("[ENSEMBLE v3] Trening zakończony.")

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Stacking meta-learner lub ważona średnia (fallback)."""
        assert self.trained, "Wywołaj train() przed predict()."

        preds = {}
        preds["xgb"] = np.clip(self.xgb_model.predict(X), 0, 1)
        preds["rf"]  = np.clip(preds["xgb"] + self.rf_model.predict(X), 0, 1)

        if self._active.get("lgb") and self.lgb_model:
            try: preds["lgb"] = np.clip(self.lgb_model.predict(X), 0, 1)
            except: pass

        if self._active.get("cb") and self.cb_model:
            try: preds["cb"] = np.clip(self.cb_model.predict(X), 0, 1)
            except: pass

        if self._active.get("ktb") and self.ktb_model:
            try: preds["ktb"] = np.clip(self.ktb_model.predict(X), 0, 1)
            except: pass

        if self._active.get("ngb") and self.ngb_model:
            try: preds["ngb"] = np.clip(self.ngb_model.predict(X), 0, 1)
            except: pass

        if self._active.get("gpb") and self.gpb_model:
            try:
                gpb_pred = self.gpb_model.predict(data=X, gp_coords_pred=X, predict_var=False)
                if isinstance(gpb_pred, dict):
                    gpb_pred = gpb_pred.get("response_mean", list(gpb_pred.values())[0])
                preds["gpb"] = np.clip(gpb_pred, 0, 1)
            except: pass

        if self.meta_learner is not None:
            # Stacking
            active_keys = [k for k in self._active if self._active[k] and k in preds and k != "rf"]
            if active_keys:
                meta_X = np.column_stack([preds[k] for k in active_keys])
                if meta_X.shape[1] == self.meta_learner.coef_.shape[0]:
                    stacked = np.clip(self.meta_learner.predict(meta_X), 0, 1)
                    return stacked

        # Fallback: ważona średnia
        all_preds = list(preds.values())
        return np.clip(np.mean(all_preds, axis=0), 0, 1)

    def predict_all_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Zwraca predykcje ze WSZYSTKICH modeli jako słownik."""
        result = {}
        result["xgb"] = np.clip(self.xgb_model.predict(X), 0, 1)
        if self._active.get("lgb") and self.lgb_model:
            try: result["lgb"] = np.clip(self.lgb_model.predict(X), 0, 1)
            except: pass
        if self._active.get("cb") and self.cb_model:
            try: result["cb"] = np.clip(self.cb_model.predict(X), 0, 1)
            except: pass
        if self._active.get("ngb") and self.ngb_model:
            try: result["ngb"] = np.clip(self.ngb_model.predict(X), 0, 1)
            except: pass
        return result

    def predict_uncertainty(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not (NGB_AVAILABLE and self.ngb_model is not None and self.trained):
            return None
        try:
            dist = self.ngb_model.pred_dist(X)
            return np.array(dist.scale)
        except Exception:
            return None

    # TIER 3: SHAP why_bet
    def explain_prediction(self, x: np.ndarray, top_n: int = 3) -> List[Dict]:
        """
        Zwraca top_n cech wpływających na predykcję z wartościami SHAP.
        Fallback: feature importances XGB.
        """
        if SHAP_AVAILABLE and self._shap_explainer is not None:
            try:
                shap_vals = self._shap_explainer.shap_values(x.reshape(1, -1))[0]
                indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
                return [
                    {
                        "feature": self._feature_names[i] if i < len(self._feature_names) else f"f{i}",
                        "shap_value": round(float(shap_vals[i]), 4),
                        "feature_value": round(float(x[i]), 4),
                    }
                    for i in indices
                ]
            except Exception:
                pass

        # Fallback: XGB feature importances
        try:
            importances = self.xgb_model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            return [
                {
                    "feature": self._feature_names[i] if i < len(self._feature_names) else f"f{i}",
                    "importance": round(float(importances[i]), 4),
                    "feature_value": round(float(x[i]), 4),
                }
                for i in indices
            ]
        except Exception:
            return []

    @property
    def tree_weights(self) -> Tuple[float, float]:
        if not self.trained:
            return (0.5, 0.5)
        total = self._xgb_importance + self._rf_importance + 1e-9
        return (self._xgb_importance / total, self._rf_importance / total)

    @property
    def available_models(self) -> List[str]:
        models = ["XGBoost", "RandomForest"]
        if self._active.get("lgb"): models.append("LightGBM")
        if self._active.get("cb"):  models.append("CatBoost")
        if self._active.get("ktb"): models.append("KTBoost")
        if self._active.get("ngb"): models.append("NGBoost")
        if self._active.get("gpb"): models.append("GPBoost")
        return models


# =============================================================================
# BANEBET DECISION ENGINE v5.4 (zachowany z v2.0)
# =============================================================================

def evaluate_banebet_v5_4_decision(model_predictions: List[float],
                                    true_label: Optional[float] = None) -> Dict:
    predictions = np.array(model_predictions)
    mean_p   = float(np.mean(predictions))
    n_models = len(predictions)
    errors   = int(sum(1 for p in predictions if p < 0.5))

    threshold_ladder = {0: 0.75, 1: 0.50, 2: 0.33, 3: 0.25, 4: 0.20, 5: 0.167, 6: 0.143}
    fraction_requirements = {
        (1,2): 0.88, (1,3): 0.75, (2,3): 0.58,
        (1,4): 0.70, (2,4): 0.53, (3,4): 0.45,
        (1,5): 0.367,(2,5): 0.497,(3,5): 0.417,(4,5): 0.633,
        (1,6): 0.333,(2,6): 0.450,(3,6): 0.383,(4,6): 0.567,(5,6): 0.733,
        (1,7): 0.300,(2,7): 0.400,(3,7): 0.350,(4,7): 0.520,(5,7): 0.680,(6,7): 0.820,
    }

    base_threshold = threshold_ladder.get(errors, 1.0)
    if mean_p < base_threshold:
        return {"action": "NO_BET", "reason": f"Mean P ({mean_p:.3f}) poniżej progu ({base_threshold:.3f})",
                "mean_p": round(mean_p, 4), "errors": errors, "n_models": n_models}

    fraction_key = (errors, n_models)
    if fraction_key in fraction_requirements:
        required_p = fraction_requirements[fraction_key]
        if mean_p < required_p:
            return {"action": "NO_BET", "reason": f"Wymóg frakcyjny {fraction_key} (wymagane {required_p:.3f})",
                    "mean_p": round(mean_p, 4), "errors": errors, "n_models": n_models}

    delta = float(np.var(predictions))
    sol_plus, sol_minus = discriminant_switch(delta, mean_p)
    gamma_correction = abs(get_negative_gamma(-0.5))
    normalized_gc    = float(np.clip(1.0 / (1.0 + abs(gamma_correction)), 0.90, 1.10))
    adjusted_confidence = float(np.clip(mean_p * normalized_gc, 0.0, 1.0))

    return {
        "action": "BET",
        "confidence": round(adjusted_confidence, 4),
        "mean_p": round(mean_p, 4),
        "error_count": errors,
        "n_models": n_models,
        "discriminant": {
            "delta": round(delta, 5),
            "sol_plus": round(sol_plus, 4),
            "sol_minus": round(sol_minus, 4) if sol_minus is not None else None,
        },
        "gamma_correction": round(normalized_gc, 4),
        "metadata": {
            "ladder_threshold": base_threshold,
            "fraction_req": fraction_requirements.get(fraction_key, "N/A"),
        },
    }


# =============================================================================
# ADAPTACYJNY UCZEŃ (zachowany z v2.0)
# =============================================================================

class AdaptiveLearner:
    def __init__(self, engine: 'UniversalBettingEngineV3', learning_rate: float = 0.05):
        self.engine = engine
        self.learning_rate = learning_rate
        self.match_history: List[Tuple[np.ndarray, float]] = []
        self.weight_corrections: Dict[str, float] = {dim: 0.0 for dim in engine.config.dimensions}
        self.bayesian_alpha: float = 20.0
        self.bayesian_beta: float = 1.0
        self.prediction_errors: List[float] = []

    def record_match(self, params: np.ndarray, actual_result: float) -> Dict:
        predicted = self.engine.predict(params)["p_win"]
        error = actual_result - predicted
        self.match_history.append((params.copy(), actual_result))
        self.prediction_errors.append(abs(error))
        self.bayesian_beta += 1.0
        corrections = {}
        for i, dim in enumerate(self.engine.config.dimensions):
            bayesian_weight = self.bayesian_beta / (self.bayesian_alpha + self.bayesian_beta)
            correction = np.clip(error * params[i] * self.learning_rate * bayesian_weight, -0.05, 0.05)
            self.weight_corrections[dim] += correction
            corrections[dim] = correction
        self._apply_corrections()
        new_predicted = self.engine.predict(params)["p_win"]
        return {
            "dimension_corrections": corrections,
            "old_prediction": round(predicted, 4),
            "actual_result": actual_result,
            "error": round(error, 4),
            "new_prediction": round(new_predicted, 4),
            "matches_learned": len(self.match_history),
        }

    def _apply_corrections(self):
        n = len(self.engine.config.dimensions)
        original_weights = np.array(self.engine.config.weights[:n])
        corrected_weights = original_weights.copy()
        for i, dim in enumerate(self.engine.config.dimensions):
            corrected_weights[i] = original_weights[i] * (1.0 + self.weight_corrections[dim])
        corrected_weights = np.maximum(corrected_weights, 0.01)
        corrected_weights /= corrected_weights.sum()
        if self.engine.mebn is not None:
            self.engine.mebn.w = corrected_weights
        self.engine._corrected_weights = corrected_weights.tolist()

    def get_current_weights(self) -> Dict[str, Dict]:
        if self.engine.mebn is None:
            return {}
        result = {}
        for i, dim in enumerate(self.engine.config.dimensions):
            original = self.engine.config.weights[i]
            result[dim] = {
                "original": original,
                "correction": round(self.weight_corrections[dim], 4),
                "current": round(self.engine.mebn.w[i], 4),
            }
        return result

    def save_memory(self, filepath: str):
        data = {
            "sport": self.engine.sport,
            "match_history": [{"params": p.tolist(), "actual_result": r} for p, r in self.match_history],
            "weight_corrections": self.weight_corrections,
            "bayesian_alpha": self.bayesian_alpha,
            "bayesian_beta": self.bayesian_beta,
            "prediction_errors": self.prediction_errors,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_memory(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.match_history = [(np.array(m["params"]), m["actual_result"]) for m in data["match_history"]]
        self.weight_corrections = data["weight_corrections"]
        self.bayesian_alpha = data["bayesian_alpha"]
        self.bayesian_beta = data["bayesian_beta"]
        self.prediction_errors = data["prediction_errors"]
        self._apply_corrections()


# =============================================================================
# GŁÓWNY SILNIK v3.0
# =============================================================================

class UniversalBettingEngineV3:
    """
    Universal Betting Engine v3.0 — BANEBET PRO

    Integruje:
    - TreeLayerV3: 9 modeli (XGB+LGB+CB+KTB+NGB+GPB+RF) + Stacking + Optuna + SHAP
    - SportMEBN: Bayesian Network
    - DixonColes: rynki OU* i CS_* (Tier 5)
    - DynamicElo: feature wejściowa (Tier 4)
    - ProbabilityCalibrator: kalibracja p_win (Tier 1)
    - AdaptiveLearner: online learning na małych zbiorach
    - BANEBET Decision v5.4: Threshold Ladder + Fractions Matrix
    """

    def __init__(self, sport: str, use_optuna: bool = True,
                 optuna_trials: int = 30,
                 calibration_method: str = "isotonic"):
        if sport not in SPORT_CONFIGS:
            available = ", ".join(SPORT_CONFIGS.keys())
            raise ValueError(f"Nieznany sport '{sport}'. Dostępne: {available}")

        self.sport = sport
        self.config = SPORT_CONFIGS[sport]
        self.tree_layer = TreeLayerV3(
            use_optuna=use_optuna,
            optuna_trials=optuna_trials,
            calibration_method=calibration_method,
        )
        self.mebn: Optional[SportMEBN] = None
        self.tt_model = None
        self._is_compiled = False
        self._corrected_weights: Optional[List[float]] = None
        self.adaptive: Optional[AdaptiveLearner] = None

        # TIER 4: Dynamiczne Elo (globalne dla sportu)
        self.elo = DynamicElo(k=32.0)

        # TIER 5: Dixon-Coles (tylko dla piłki nożnej)
        self.dixon_coles = DixonColes() if sport == "football" else None

        n_dim = len(self.config.dimensions)
        print(f"\n[ENGINE v3.0] Inicjalizacja: {self.config.name} ({n_dim} wymiarów)")
        print(f"[ENGINE v3.0] LGB={LGB_AVAILABLE} | CB={CB_AVAILABLE} | OPTUNA={OPTUNA_AVAILABLE} | SHAP={SHAP_AVAILABLE}")

    # ------------------------------------------------------------------
    # TRENING
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
        print(f"[TRAIN v3.0] Trenowanie ensemble (K={n_splits})...")
        self.tree_layer.train(
            X, y, n_splits=n_splits,
            feature_names=self.config.dimensions,
        )
        xw, rw = self.tree_layer.tree_weights
        self.mebn = SportMEBN(self.config, tree_weights=(xw, rw))
        self.adaptive = AdaptiveLearner(self)
        print("[TRAIN v3.0] Gotowy.")

    # ------------------------------------------------------------------
    # KOMPILACJA TT
    # ------------------------------------------------------------------

    def compile(self, n_points: int = 15, eps: float = 1e-4) -> None:
        if self.mebn is None:
            raise RuntimeError("Wywołaj train() przed compile().")
        n = len(self.config.dimensions)
        if XFAC_AVAILABLE:
            self.tt_model = xfacpy.cross(
                self.mebn.probability_function,
                np.zeros(n), np.ones(n),
                n_points=n_points, eps=eps,
            )
            self._is_compiled = True
            print("[COMPILE] Tensor Train gotowy.")
        else:
            print("[COMPILE] xfacpy niedostępne — MEBN bezpośrednio.")

    # ------------------------------------------------------------------
    # PREDYKCJA
    # ------------------------------------------------------------------

    def predict(self, match_params: np.ndarray,
                team_home: Optional[str] = None,
                team_away: Optional[str] = None) -> Dict:
        """
        Predykcja dla jednego meczu.

        match_params: wektor [0,1] dla każdego wymiaru SportConfig.
        team_home, team_away: opcjonalne nazwy drużyn dla Elo feature.
        """
        if self.mebn is None:
            raise RuntimeError("Wywołaj train() przed predict().")

        n = len(self.config.dimensions)
        if len(match_params) != n:
            raise ValueError(f"Oczekiwano {n} parametrów, otrzymano {len(match_params)}.")

        # TIER 4: Dynamiczne Elo jako dodatkowa feature (nie zastępuje, ale uzupełnia)
        elo_feature = 0.5
        if team_home and team_away:
            elo_feature = self.elo.elo_feature(team_home, team_away)
            # Zastąp pierwszy wymiar (team_power) wartością Elo
            match_params = match_params.copy()
            match_params[0] = elo_feature

        # Predykcja MEBN/TT
        if self._is_compiled and self.tt_model is not None:
            p_win_raw = float(self.tt_model.eval(match_params))
        else:
            p_win_raw = self.mebn.probability_function(match_params)

        # TIER 1: Kalibracja p_win
        p_win = self.tree_layer.calibrator.calibrate(p_win_raw)

        # Obliczanie rynków podstawowych
        results = self._calculate_markets(p_win, match_params)

        # TIER 5: Dixon-Coles fusion dla rynków OU* i CS_* (tylko football)
        if self.dixon_coles is not None and self.config.attack_idx is not None:
            attack_h = float(match_params[self.config.attack_idx])
            defense_h = float(match_params[self.config.defense_idx]) if self.config.defense_idx else 0.5
            # Odwrócone dla gości
            attack_a = 1.0 - defense_h
            defense_a = 1.0 - attack_h
            home_adv = float(match_params[1]) if len(match_params) > 1 else 1.0

            self.dixon_coles.fit_from_params(attack_h, defense_h, attack_a, defense_a, home_adv + 0.7)

            # Podmień OU* predykcje na Dixon-Coles
            dc_ou = self.dixon_coles.over_under_probs()
            results.update({k: v for k, v in dc_ou.items()})

            # Podmień CS_* na Monte Carlo
            mc_cs = monte_carlo_correct_score(
                self.dixon_coles.mu_h, self.dixon_coles.mu_a, n_sims=10000
            )
            results.update(mc_cs)
            results["dc_mu_h"] = round(self.dixon_coles.mu_h, 3)
            results["dc_mu_a"] = round(self.dixon_coles.mu_a, 3)
            results["p_btts_yes"] = round(self.dixon_coles.btts_prob(), 4)
            results["p_btts_no"]  = round(1.0 - results["p_btts_yes"], 4)

        # TIER 3: SHAP why_bet
        why_bet = []
        if self.tree_layer.trained:
            why_bet = self.tree_layer.explain_prediction(match_params, top_n=3)
        results["why_bet"] = why_bet

        # Niepewność NGBoost
        uncertainty = self.predict_uncertainty(match_params)
        if uncertainty is not None:
            results["ngb_uncertainty_std"] = round(uncertainty, 4)

        # Elo feature
        results["elo_feature"] = round(elo_feature, 4)

        # Adaptive weights info
        if self.adaptive and self._corrected_weights:
            results["adaptive_weights_active"] = True

        # Kalibracja info
        results["p_win_raw"] = round(p_win_raw, 4)
        results["p_win_calibrated"] = round(p_win, 4)

        # BANEBET v5.4
        ensemble_preds = self._collect_ensemble_predictions(match_params, p_win)
        results["banebet_v54"] = evaluate_banebet_v5_4_decision(ensemble_preds)

        return results

    def predict_batch(self, batch: np.ndarray) -> List[Dict]:
        return [self.predict(row) for row in batch]

    def predict_uncertainty(self, match_params: np.ndarray) -> Optional[float]:
        if self.mebn is None:
            return None
        X = match_params.reshape(1, -1)
        std_arr = self.tree_layer.predict_uncertainty(X)
        return float(std_arr[0]) if std_arr is not None else None

    # ------------------------------------------------------------------
    # ZBIERANIE PREDYKCJI DLA BANEBET
    # ------------------------------------------------------------------

    def _collect_ensemble_predictions(self, match_params: np.ndarray, p_win_mebn: float) -> List[float]:
        preds = [p_win_mebn]
        if not self.tree_layer.trained:
            return preds * 7

        X = match_params.reshape(1, -1)
        model_preds = self.tree_layer.predict_all_models(X)
        for key, arr in model_preds.items():
            preds.append(float(arr[0]))

        while len(preds) < 2:
            preds.append(p_win_mebn)
        return preds

    # ------------------------------------------------------------------
    # RYNKI BUKMACHERSKIE
    # ------------------------------------------------------------------

    def _calculate_markets(self, p_win: float, params: np.ndarray) -> Dict:
        p_win  = np.clip(p_win, 0.01, 0.99)
        p_draw = self._estimate_draw(p_win, params)
        p_loss = max(0.01, 1.0 - p_win - p_draw)

        total  = p_win + p_draw + p_loss
        p_win  /= total
        p_draw /= total
        p_loss /= total

        fair_1 = 1.0 / p_win
        fair_x = 1.0 / p_draw if p_draw > 0.005 else None
        fair_2 = 1.0 / p_loss

        fatigue_idx = self._dim_index("fatigue", "back2back", "tiredness")
        form_idx    = self._dim_index("form", "form_last5", "current_form")
        fatigue     = float(params[fatigue_idx]) if fatigue_idx is not None else 0.3
        form        = float(params[form_idx])    if form_idx    is not None else 0.5

        p_over = self._estimate_over(p_win, fatigue, form)
        p_btts = self._estimate_btts(p_over, p_draw)

        _bookie_odds = (1.0 / p_win) * 1.05
        _b = _bookie_odds - 1.0
        _q = 1.0 - p_win
        kelly_full = (p_win * _b - _q) / (_b + 1e-9)
        kelly_frac = max(0.0, kelly_full / 8.0)

        out = {
            "sport": self.config.name,
            "dimensions": dict(zip(self.config.dimensions, params.tolist())),
            "p_win": round(p_win, 4),
            "p_draw": round(p_draw, 4),
            "p_loss": round(p_loss, 4),
            "fair_1": round(fair_1, 3),
            "fair_x": round(fair_x, 3) if fair_x else None,
            "fair_2": round(fair_2, 3),
            "p_over": round(p_over, 4),
            "p_under": round(1 - p_over, 4),
            "fair_over": round(1.0 / p_over, 3),
            "fair_under": round(1.0 / (1 - p_over), 3),
            "p_btts_yes": round(p_btts, 4),
            "p_btts_no":  round(1 - p_btts, 4),
            "kelly_fraction": round(kelly_frac, 4),
            "available_markets": self.config.markets,
            "confidence": self._confidence_label(p_win),
        }
        out["p_dc_1x"] = round(p_win + p_draw, 4)
        out["p_dc_x2"] = round(p_draw + p_loss, 4)
        out["p_dc_12"] = round(p_win + p_loss, 4)
        return out

    def _estimate_draw(self, p_win: float, params: np.ndarray) -> float:
        sport_draw_base = {
            "football": 0.26, "hockey": 0.00, "basketball": 0.00,
            "tennis": 0.00, "volleyball": 0.00, "baseball": 0.00,
            "american_football": 0.005, "rugby": 0.02, "cycling": 0.00,
            "boxing_mma": 0.05, "snooker": 0.00, "darts": 0.00,
            "esports": 0.00, "racing": 0.00, "cricket": 0.20,
        }
        base = sport_draw_base.get(self.sport, 0.0)
        imbalance = abs(p_win - 0.5) * 2.0
        draw_scale = float(np.clip(1.0 - 0.60 * imbalance, 0.20, 1.0))
        return float(np.clip(base * draw_scale, 0.0, base))

    def _estimate_over(self, p_win: float, fatigue: float, form: float) -> float:
        return float(np.clip(0.52 + form * 0.10 - fatigue * 0.08, 0.30, 0.75))

    def _estimate_btts(self, p_over: float, p_draw: float) -> float:
        return float(np.clip(p_over * 0.7 + p_draw * 0.5, 0.20, 0.80))

    def _dim_index(self, *names: str) -> Optional[int]:
        for name in names:
            if name in self.config.dimensions:
                return self.config.dimensions.index(name)
        return None

    def _confidence_label(self, p_win: float) -> str:
        if p_win >= 0.80: return "BARDZO MOCNY ZAKŁAD ★★★★★"
        if p_win >= 0.75: return "MOCNY ZAKŁAD ★★★★★ [PRÓG]"
        if p_win >= 0.68: return "DOBRY ZAKŁAD ★★★★"
        if p_win >= 0.58: return "ZAKŁAD ★★★"
        if p_win >= 0.50: return "SŁABY ZAKŁAD ★★"
        return "UNIKAJ ★"

    def kelly_with_odds(self, p_win: float, real_odds: float, kelly_divisor: float = 8.0) -> float:
        if real_odds <= 1.0 or p_win <= 0:
            return 0.0
        b = real_odds - 1.0
        q = 1.0 - p_win
        kelly_full = (p_win * b - q) / (b + 1e-9)
        if kelly_full <= 0:
            return 0.0
        return round(float(np.clip(kelly_full / kelly_divisor, 0.0, 0.25)), 4)

    # ------------------------------------------------------------------
    # ADAPTACYJNE UCZENIE
    # ------------------------------------------------------------------

    def record_match_result(self, params: np.ndarray, actual_result: float) -> Dict:
        if self.adaptive is None:
            raise RuntimeError("Wywołaj train() przed record_match_result().")
        return self.adaptive.record_match(params, actual_result)

    def update_elo(self, team_home: str, team_away: str, score_home: float) -> Tuple[float, float]:
        """
        Aktualizuje rankingi Elo po meczu.
        score_home: 1=wygrana domu, 0.5=remis, 0=przegrana domu
        """
        return self.elo.update(team_home, team_away, score_home)

    def get_adaptive_weights(self) -> Dict:
        if self.adaptive is None:
            return {}
        return self.adaptive.get_current_weights()

    def save_adaptive_memory(self, filepath: str = None):
        if self.adaptive is None:
            return
        filepath = filepath or f"adaptive_memory_v3_{self.sport}.json"
        self.adaptive.save_memory(filepath)

    def load_adaptive_memory(self, filepath: str = None):
        if self.adaptive is None:
            return
        filepath = filepath or f"adaptive_memory_v3_{self.sport}.json"
        self.adaptive.load_memory(filepath)

    def ensemble_info(self) -> Dict:
        info = {
            "version": "3.0",
            "available_models": self.tree_layer.available_models if self.tree_layer.trained else [],
            "stacking_meta_learner": self.tree_layer.meta_learner is not None if self.tree_layer.trained else False,
            "calibration": self.tree_layer.calibrator.method,
            "calibration_fitted": self.tree_layer.calibrator.fitted,
            "optuna_used": self.tree_layer.use_optuna,
            "shap_available": SHAP_AVAILABLE,
            "xfac_compiled": self._is_compiled,
            "dixon_coles": self.dixon_coles is not None,
            "dynamic_elo_teams": len(self.elo.ratings),
            "modules": {
                "XGBoost": XGB_AVAILABLE,
                "LightGBM": LGB_AVAILABLE,
                "CatBoost": CB_AVAILABLE,
                "NGBoost": NGB_AVAILABLE,
                "GPBoost": GPB_AVAILABLE,
                "KTBoost": KTB_AVAILABLE,
                "Optuna": OPTUNA_AVAILABLE,
                "SHAP": SHAP_AVAILABLE,
                "XFAC": XFAC_AVAILABLE,
            },
        }
        if self.adaptive:
            info["adaptive"] = {
                "matches_learned": len(self.adaptive.match_history),
                "avg_error_last10": round(np.mean(self.adaptive.prediction_errors[-10:]), 4) if self.adaptive.prediction_errors else None,
            }
        return info

    def describe(self) -> None:
        print(f"\n{'='*65}")
        print(f"  UNIVERSAL BETTING ENGINE v3.0 — BANEBET PRO")
        print(f"  Sport: {self.config.name.upper()}")
        print(f"{'='*65}")
        print(f"  TIER 1: Ensemble + Kalibracja")
        print(f"    XGBoost:   {'✓' if XGB_AVAILABLE else '✗'}")
        print(f"    LightGBM:  {'✓' if LGB_AVAILABLE else '✗ (pip install lightgbm)'}  ← NOWY")
        print(f"    CatBoost:  {'✓' if CB_AVAILABLE  else '✗ (pip install catboost)'}  ← NOWY")
        print(f"    KTBoost:   {'✓' if KTB_AVAILABLE else '✗'}")
        print(f"    NGBoost:   {'✓' if NGB_AVAILABLE else '✗'}")
        print(f"    GPBoost:   {'✓' if GPB_AVAILABLE else '✗'}")
        print(f"    RF:        ✓ (residual corrector)")
        print(f"    Kalibracja:{self.tree_layer.calibrator.method.upper()}  ← NOWA")
        print(f"\n  TIER 2: Stacking meta-learner Ridge na OOF  ← NOWY")
        print(f"\n  TIER 3: HPO + Explainability")
        print(f"    Optuna HPO: {'✓' if OPTUNA_AVAILABLE else '✗ (pip install optuna)'}  ← NOWY")
        print(f"    SHAP why_bet: {'✓' if SHAP_AVAILABLE else '✗ (pip install shap)'}  ← NOWY")
        print(f"\n  TIER 4: Dynamic Features")
        print(f"    Dynamiczne Elo K=32: ✓  ← NOWY")
        print(f"    Monte Carlo CS (10k sims): ✓  ← NOWY")
        print(f"\n  TIER 5: Dixon-Coles fusion (OU/CS): {'✓' if self.dixon_coles else '—'}  ← NOWY")
        print(f"\n  CORE: MEBN + XFAC TT: {'✓' if XFAC_AVAILABLE else '—'}")
        print(f"  CORE: BANEBET Decision v5.4: ✓")
        print(f"  CORE: AdaptiveLearner: ✓")
        print(f"  CORE: DynamicElo: ✓")
        print(f"{'='*65}\n")


# =============================================================================
# FABRYKA
# =============================================================================

def create_engine(sport: str, use_optuna: bool = True, optuna_trials: int = 30) -> UniversalBettingEngineV3:
    """Tworzy silnik v3.0 dla danego sportu."""
    return UniversalBettingEngineV3(sport, use_optuna=use_optuna, optuna_trials=optuna_trials)

def list_sports() -> List[str]:
    return list(SPORT_CONFIGS.keys())


# =============================================================================
# DEMO
# =============================================================================

def demo_single(sport: str = "football", use_optuna: bool = False):
    """
    Pełne demo dla jednego sportu.
    use_optuna=False → szybki demo bez HPO (HPO można włączyć produkcyjnie).
    """
    engine = create_engine(sport, use_optuna=use_optuna, optuna_trials=20)
    engine.describe()

    config = SPORT_CONFIGS[sport]
    n_dim = len(config.dimensions)

    # Dane syntetyczne
    np.random.seed(42)
    n = 800
    X = np.random.rand(n, n_dim)
    w = np.array(config.weights[:n_dim])
    w /= w.sum()
    y = np.clip(X @ w + np.random.randn(n) * 0.05, 0, 1)

    engine.train(X, y, n_splits=3)
    engine.compile(n_points=10, eps=1e-3)

    # Przykładowy mecz
    params = np.array([0.80, 0.75, 0.30, 0.70, 0.60, 0.55, 0.45, 0.55][:n_dim])

    # Opcjonalne: aktualizacja Elo
    if sport == "football":
        engine.update_elo("TeamA", "TeamB", 1.0)  # TeamA wygrała 5 poprzednich
        engine.update_elo("TeamA", "TeamC", 1.0)
        engine.update_elo("TeamA", "TeamD", 0.5)

    result = engine.predict(params, team_home="TeamA", team_away="TeamB")

    print("\n  === WYNIK PREDYKCJI v3.0 ===")
    print(f"  p_win (raw)       : {result['p_win_raw']}")
    print(f"  p_win (calibrated): {result['p_win_calibrated']}")
    print(f"  p_draw            : {result['p_draw']}")
    print(f"  p_loss            : {result['p_loss']}")
    print(f"  Elo feature       : {result['elo_feature']}")
    print(f"  Kelly fraction    : {result['kelly_fraction']}")
    print(f"  Confidence        : {result['confidence']}")

    if sport == "football":
        print(f"\n  === DIXON-COLES (OU/CS) ===")
        print(f"  mu_H={result.get('dc_mu_h')}, mu_A={result.get('dc_mu_a')}")
        for k in ["OU15", "OU25", "OU35"]:
            print(f"  P({k})={result.get(k, 'N/A')}")
        print(f"  CS_10={result.get('CS_10', 'N/A')}, CS_11={result.get('CS_11', 'N/A')}, CS_21={result.get('CS_21', 'N/A')}")
        print(f"  BTTS_YES={result.get('p_btts_yes', 'N/A')}")

    print(f"\n  === WHY BET (SHAP/Importance) ===")
    for feat in result.get("why_bet", []):
        print(f"  {feat}")

    print(f"\n  === BANEBET v5.4 ===")
    bd = result.get("banebet_v54", {})
    print(f"  Decyzja: {bd.get('action')} | Confidence: {bd.get('confidence', 'N/A')}")

    print(f"\n  === ENSEMBLE INFO ===")
    print(json.dumps(engine.ensemble_info(), ensure_ascii=False, indent=2))


def demo_all_sports(use_optuna: bool = False):
    """Demo szybkie dla wszystkich sportów."""
    print("\n" + "="*70)
    print("  UNIVERSAL BETTING ENGINE v3.0 — ALL SPORTS DEMO")
    print("="*70)

    for sport_key in SPORT_CONFIGS:
        config = SPORT_CONFIGS[sport_key]
        n_dim = len(config.dimensions)

        print(f"\n  {'─'*60}")
        print(f"  {config.name.upper()}")
        print(f"  {'─'*60}")

        engine = UniversalBettingEngineV3(sport_key, use_optuna=False)

        np.random.seed(42)
        n_samples = 400
        X_hist = np.random.rand(n_samples, n_dim)
        w = np.array(config.weights[:n_dim]); w /= w.sum()
        y_hist = np.clip(X_hist @ w + np.random.randn(n_samples) * 0.05, 0, 1)

        engine.train(X_hist, y_hist, n_splits=3)
        engine.compile(n_points=8, eps=1e-3)

        live_params = np.random.rand(n_dim)
        result = engine.predict(live_params)

        print(f"  p_win={result['p_win_calibrated']:.4f}  p_draw={result['p_draw']:.4f}  p_loss={result['p_loss']:.4f}")
        print(f"  Kelly={result['kelly_fraction']:.4f} | {result['confidence']}")
        print(f"  Banebet: {result['banebet_v54'].get('action')}")
        if result.get("why_bet"):
            top = result["why_bet"][0]
            print(f"  Top feature: {top.get('feature', top.get('feature', '?'))} = {top.get('feature_value', '?')}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sport_arg = sys.argv[1].lower()
        optuna_flag = "--optuna" in sys.argv

        if sport_arg == "all":
            demo_all_sports(use_optuna=False)
        elif sport_arg in SPORT_CONFIGS:
            demo_single(sport_arg, use_optuna=optuna_flag)
        else:
            print(f"Nieznany sport. Dostępne: {list_sports()}")
    else:
        demo_single("football", use_optuna=False)
        print("\n\nAby włączyć Optuna HPO: python betting_model_3_0.py football --optuna")
        print(f"Wszystkie sporty: python betting_model_3_0.py all")
        print(f"Dostępne sporty: {', '.join(list_sports())}")