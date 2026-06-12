"""
=============================================================================
HYBRID BETTING MODEL v2.0 — BANEBET PRO
=============================================================================
POTWOR: Rozszerzenie Universal Betting Engine v4.0, laczace WSZYSTKIE
poprzednie etapy wiedzy (TIER 1-5) z nowa warstwa hybrydowa (TIER 6).
NICZEGO Z v4.0 NIE USUNIETO -- calosc v4.0 jest zachowana i aktywna.

ZACHOWANE Z v4.0 (TIER 1-5):
  TIER 1: LightGBM + CatBoost + Kalibracja Isotonic/Platt (9 modeli ensemble)
  TIER 2: Stacking meta-learner (Ridge) zamiast wazonej MAE
  TIER 3: Optuna HPO + SHAP (pole why_bet w odpowiedzi)
  TIER 4: Dynamiczne Elo jako feature + Monte Carlo Correct Score
  TIER 5: Dixon-Coles fusion dla rynkow OU* i CS_*

NOWE W v1.0 HYBRID (TIER 6) -- 9 modulow dobudowanych na wierzchu:
  TIER 6.1: Market-Based Calibration (MBC)
  TIER 6.2: Temporal Attention Mechanism
  TIER 6.3: Injury/Suspension Impact Model
  TIER 6.4: Weather & Pitch Impact Module
  TIER 6.5: Pseudo-labeling + Self-training
  TIER 6.6: Contrastive Learning na embeddingach druzyn
  TIER 6.7: Dynamic Threshold Tuning
  TIER 6.8: GPU Acceleration + Batch Inference
  TIER 6.9: Multi-Loss Optimization

GLOWNA KLASA: HybridBettingEngineV1 (dziedziczy po UniversalBettingEngineV3)
FABRYKA: create_hybrid_engine(sport, ...)

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
    XFAC_AVAILABLE = hasattr(xfacpy, 'TensorCI2')
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
                "min_data_in_leaf": 5, "objective": "regression_l2", "verbose": -1,
            }

        # RandomForest
        self.rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=8, criterion="squared_error", bootstrap=True, n_jobs=-1, random_state=42,
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
                    self.lgb_model.fit(np.array(Xt), yt)
                    oof_dict["lgb"][val_idx] = np.clip(self.lgb_model.predict(np.array(Xv)), 0, 1)
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
            try: self.lgb_model.fit(np.array(X), y)
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
            try: preds["lgb"] = np.clip(self.lgb_model.predict(np.array(X)), 0, 1)
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
            try: result["lgb"] = np.clip(self.lgb_model.predict(np.array(X)), 0, 1)
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
    Universal Betting Engine v4.0 — BANEBET PRO

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
        print(f"\n[ENGINE v4.0] Inicjalizacja: {self.config.name} ({n_dim} wymiarów)")
        print(f"[ENGINE v4.0] LGB={LGB_AVAILABLE} | CB={CB_AVAILABLE} | OPTUNA={OPTUNA_AVAILABLE} | SHAP={SHAP_AVAILABLE}")

    # ------------------------------------------------------------------
    # TRENING
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
        print(f"[TRAIN v4.0] Trenowanie ensemble (K={n_splits})...")
        self.tree_layer.train(
            X, y, n_splits=n_splits,
            feature_names=self.config.dimensions,
        )
        xw, rw = self.tree_layer.tree_weights
        self.mebn = SportMEBN(self.config, tree_weights=(xw, rw))
        self.adaptive = AdaptiveLearner(self)
        print("[TRAIN v4.0] Gotowy.")

    # ------------------------------------------------------------------
    # KOMPILACJA TT
    # ------------------------------------------------------------------

    def compile(self, n_points: int = 15, eps: float = 1e-4) -> None:
        if self.mebn is None:
            raise RuntimeError("Wywołaj train() przed compile().")
        n = len(self.config.dimensions)
        if XFAC_AVAILABLE:
            try:
                grid = [np.linspace(0, 1, n_points) for _ in range(n)]
                param = xfacpy.TensorCI2Param()
                param.reltol = eps
                tci = xfacpy.TensorCI2(self.mebn.probability_function, grid, param)
                self.tt_model = tci.tt
                self._is_compiled = True
                print("[COMPILE] Tensor Train gotowy (TensorCI2).")
            except Exception as e:
                warnings.warn(f"[COMPILE] xfacpy błąd: {e} — MEBN bezpośrednio.")
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
            try:
                p_win_raw = float(self.tt_model(match_params.tolist()))
            except Exception:
                p_win_raw = self.mebn.probability_function(match_params)
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
            "version": "4.0",
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
        print(f"  UNIVERSAL BETTING ENGINE v4.0 — BANEBET PRO")
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
    print("  UNIVERSAL BETTING ENGINE v4.0 — ALL SPORTS DEMO")
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
# =============================================================================
#   HYBRID BETTING MODEL v1.0 — BANEBET PRO
#   ROZSZERZENIE v4.0 → HYBRID v1.0
#
#   Poniżej 9 nowych modułów dodanych BEZ USUWANIA niczego z v4.0:
#     1. Market-Based Calibration (MBC)
#     2. Temporal Attention Mechanism (forma w czasie)
#     3. Injury/Suspension Impact Model
#     4. Weather & Pitch Impact Module
#     5. Pseudo-labeling + Self-training
#     6. Contrastive Learning na embeddingach drużyn
#     7. Dynamic Threshold Tuning (adaptacyjny próg BANEBET)
#     8. GPU Acceleration + Batch Inference (config helper)
#     9. Multi-Loss Optimization (ranking + calibration + AUC)
#
#   Wszystko zintegrowane w HybridBettingEngineV1, który DZIEDZICZY
#   po UniversalBettingEngineV3 (v4.0) — nic z v4.0 nie jest usuwane,
#   tylko nadbudowane.
# =============================================================================
# =============================================================================

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from collections import deque

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False


# =============================================================================
# MODULE 1: MARKET-BASED CALIBRATION (MBC)
# =============================================================================

class MarketBasedCalibrator:
    """
    Korekta p_win na podstawie implied probability z kursów bukmacherskich
    (10+ bukmacherów). Wykrywa value bets przez porównanie modelu vs rynku.

    Implied probability liczona jest z marginesu (overround) usuniętym,
    a finalne p_win jest mieszanką modelu i rynku (blend), co redukuje
    overfitting modelu do syntetycznych/historycznych danych.
    """

    def __init__(self, blend_weight: float = 0.35, min_bookies: int = 3):
        """
        blend_weight: waga rynku w finalnej mieszance (0=ignoruj rynek, 1=tylko rynek)
        min_bookies: minimalna liczba kursów do uznania danych rynkowych za wiarygodne
        """
        self.blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
        self.min_bookies = min_bookies
        self.history: List[Dict] = []

    @staticmethod
    def odds_to_implied_prob(odds: float) -> float:
        """Surowe implied probability z kursu dziesiętnego."""
        if odds <= 1.0:
            return 0.0
        return float(np.clip(1.0 / odds, 0.0, 1.0))

    def remove_overround(self, odds_1: List[float], odds_x: List[float],
                          odds_2: List[float]) -> Dict[str, float]:
        """
        Usuwa margines bukmacherski (overround) z kursów wielu bukmacherów,
        zwraca uśrednione, znormalizowane implied probabilities dla 1/X/2.
        """
        n = min(len(odds_1), len(odds_x), len(odds_2))
        if n == 0:
            return {"p1": 1/3, "px": 1/3, "p2": 1/3, "overround_avg": 1.0, "n_bookies": 0}

        p1_list, px_list, p2_list, overrounds = [], [], [], []
        for i in range(n):
            p1_raw = self.odds_to_implied_prob(odds_1[i])
            px_raw = self.odds_to_implied_prob(odds_x[i])
            p2_raw = self.odds_to_implied_prob(odds_2[i])
            total = p1_raw + px_raw + p2_raw
            if total <= 0:
                continue
            overrounds.append(total)
            p1_list.append(p1_raw / total)
            px_list.append(px_raw / total)
            p2_list.append(p2_raw / total)

        if not p1_list:
            return {"p1": 1/3, "px": 1/3, "p2": 1/3, "overround_avg": 1.0, "n_bookies": 0}

        return {
            "p1": float(np.mean(p1_list)),
            "px": float(np.mean(px_list)),
            "p2": float(np.mean(p2_list)),
            "overround_avg": float(np.mean(overrounds)),
            "n_bookies": len(p1_list),
            "p1_std": float(np.std(p1_list)),
            "px_std": float(np.std(px_list)),
            "p2_std": float(np.std(p2_list)),
        }

    def calibrate_with_market(self, p_model: float, market_probs: Dict[str, float],
                               outcome_key: str = "p1") -> Dict:
        """
        Łączy p_model (z ensemble/MEBN) z implied probability rynku.
        Zwraca p_blended + ocenę value (edge).
        """
        p_market = market_probs.get(outcome_key, p_model)
        n_bookies = market_probs.get("n_bookies", 0)

        # Jeśli mało bukmacherów — zmniejsz wagę rynku proporcjonalnie
        effective_weight = self.blend_weight
        if n_bookies < self.min_bookies:
            effective_weight *= (n_bookies / max(self.min_bookies, 1))

        p_blended = float(np.clip(
            (1.0 - effective_weight) * p_model + effective_weight * p_market,
            0.01, 0.99
        ))

        edge = p_model - p_market  # >0 => model widzi value (model bardziej optymistyczny niż rynek)

        value_rating = "NONE"
        if edge > 0.05:
            value_rating = "STRONG_VALUE"
        elif edge > 0.02:
            value_rating = "VALUE"
        elif edge < -0.05:
            value_rating = "AVOID_TRAP"

        return {
            "p_model": round(p_model, 4),
            "p_market": round(p_market, 4),
            "p_blended": round(p_blended, 4),
            "edge": round(edge, 4),
            "value_rating": value_rating,
            "n_bookies": n_bookies,
            "effective_market_weight": round(effective_weight, 4),
        }

    def record(self, result: Dict):
        self.history.append(result)


# =============================================================================
# MODULE 2: TEMPORAL ATTENTION MECHANISM (forma drużyny w czasie)
# =============================================================================

class TemporalAttentionForm:
    """
    Lekka implementacja mechanizmu uwagi czasowej dla ostatnich N meczów
    drużyny — bez wymagania pełnego frameworku DL (działa na numpy).

    Każdy mecz ma wektor cech (np. [wynik, gole_strzelone, gole_stracone,
    xG, forma_rywala]). Mechanizm uwagi przypisuje wagi nowszym meczom
    silniej, ale moduluje je przez "query" (kontekst aktualnego meczu),
    co pozwala wychwycić momentum, cykle formy i efekt zmęczenia kumulacyjnego.

    To jest self-attention w stylu Transformera (Q,K,V) zaimplementowany
    w czystym numpy — bez torch/tensorflow, więc działa zawsze.
    """

    def __init__(self, window: int = 20, feature_dim: int = 5,
                 temperature: float = 1.0):
        self.window = window
        self.feature_dim = feature_dim
        self.temperature = temperature

        # Proste, trenowalne projekcje Q/K/V (inicjalizacja losowa, uczone
        # online metodą gradientu na błędzie predykcji formy)
        rng = np.random.RandomState(42)
        self.W_q = rng.normal(0, 0.1, (feature_dim, feature_dim))
        self.W_k = rng.normal(0, 0.1, (feature_dim, feature_dim))
        self.W_v = rng.normal(0, 0.1, (feature_dim, feature_dim))
        self.lr = 0.01

        # Historie per drużyna: deque z max długości window
        self.team_histories: Dict[str, deque] = {}

    def add_match(self, team: str, feature_vector: np.ndarray):
        """Dodaje wektor cech meczu do historii drużyny (np. po zakończeniu meczu)."""
        fv = np.asarray(feature_vector, dtype=float)
        if fv.shape[0] != self.feature_dim:
            # Padding / truncation dla bezpieczeństwa
            if fv.shape[0] < self.feature_dim:
                fv = np.pad(fv, (0, self.feature_dim - fv.shape[0]))
            else:
                fv = fv[:self.feature_dim]
        if team not in self.team_histories:
            self.team_histories[team] = deque(maxlen=self.window)
        self.team_histories[team].append(fv)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x / max(self.temperature, 1e-6))
        return e / (np.sum(e) + 1e-12)

    def attended_form(self, team: str, current_context: Optional[np.ndarray] = None) -> Dict:
        """
        Zwraca skalar 'attended_form_score' [0,1] dla drużyny, uwzględniający
        momentum (trend ostatnich meczów) oraz kontekst aktualnego meczu
        (np. siła rywala) jako query.

        Jeśli historia drużyny jest pusta, zwraca neutralne 0.5.
        """
        hist = self.team_histories.get(team)
        if not hist or len(hist) == 0:
            return {"attended_form_score": 0.5, "momentum": 0.0, "n_matches": 0}

        H = np.stack(list(hist))  # (T, feature_dim)
        T = H.shape[0]

        if current_context is None:
            query_vec = H[-1]
        else:
            cc = np.asarray(current_context, dtype=float)
            if cc.shape[0] != self.feature_dim:
                if cc.shape[0] < self.feature_dim:
                    cc = np.pad(cc, (0, self.feature_dim - cc.shape[0]))
                else:
                    cc = cc[:self.feature_dim]
            query_vec = cc

        # Self-attention: Q z query_vec, K i V z historii
        Q = query_vec @ self.W_q                 # (feature_dim,)
        K = H @ self.W_k                          # (T, feature_dim)
        V = H @ self.W_v                          # (T, feature_dim)

        scores = K @ Q / np.sqrt(self.feature_dim)  # (T,)

        # Recency bias: nowsze meczy (wyższy indeks T) dostają dodatkowy bonus
        recency_bonus = np.linspace(-1.0, 1.0, T) * 0.5
        scores = scores + recency_bonus

        attn = self._softmax(scores)  # (T,)
        context = attn @ V            # (feature_dim,)

        # Zakładamy że feature[0] = wynik meczu (1=win, 0.5=draw, 0=loss)
        attended_form_score = float(np.clip(expit_local(context[0]), 0.0, 1.0))

        # Momentum: różnica między średnią z drugiej i pierwszej połowy historii
        half = max(1, T // 2)
        recent_avg = float(np.mean(H[-half:, 0]))
        older_avg = float(np.mean(H[:T - half, 0])) if T - half > 0 else recent_avg
        momentum = float(np.clip(recent_avg - older_avg, -1.0, 1.0))

        return {
            "attended_form_score": round(attended_form_score, 4),
            "momentum": round(momentum, 4),
            "n_matches": T,
            "attention_weights": [round(float(a), 4) for a in attn],
        }

    def online_update(self, team: str, predicted_form: float, actual_result: float):
        """
        Prosty gradient update Q/K/V projekcji na podstawie błędu
        predykcji formy vs rzeczywisty wynik (semi-supervised refinement).
        """
        error = actual_result - predicted_form
        # Bardzo lekka korekta — celem jest stabilność, nie pełny backprop
        grad_scale = np.clip(error * self.lr, -0.01, 0.01)
        self.W_q += grad_scale * np.eye(self.feature_dim)
        self.W_k += grad_scale * np.eye(self.feature_dim) * 0.5
        self.W_v += grad_scale * np.eye(self.feature_dim) * 0.5


def expit_local(x):
    """Lokalna kopia sigmoid, niezależna od scipy importu, dla bezpieczeństwa modułu."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# =============================================================================
# MODULE 3: INJURY / SUSPENSION IMPACT MODEL
# =============================================================================

@dataclass
class PlayerAbsence:
    player_name: str
    position: str  # "GK", "DEF", "MID", "FWD"
    importance: float = 0.5  # [0,1] - waga zawodnika w zespole (np. % minut, rating)
    reason: str = "injury"   # "injury" lub "suspension"


class InjurySuspensionImpactModel:
    """
    Bayesowski model wpływu nieobecności zawodników na p_win.

    Każda pozycja ma bazowy mnożnik wpływu — utrata napastnika boli bardziej
    niż utrata obrońcy. Wpływ skaluje się przez 'importance' zawodnika
    (gwiazda vs rezerwowy).

    Output: injury_impact_home / injury_impact_away — wartości ujemne
    [-0.12, 0.0] reprezentujące spadek p_win drużyny.
    """

    POSITION_BASE_IMPACT = {
        "GK":  0.06,
        "DEF": 0.04,
        "MID": 0.07,
        "FWD": 0.09,
    }

    MAX_TOTAL_IMPACT = 0.12  # górny limit spadku p_win z tytułu absencji

    def __init__(self):
        pass

    def compute_impact(self, absences: List[PlayerAbsence]) -> Dict:
        """
        Zwraca słownik z impact_score (ujemny, [-0.12, 0]) oraz
        rozbiciem per zawodnik.
        """
        if not absences:
            return {"impact_score": 0.0, "details": [], "n_absences": 0}

        total = 0.0
        details = []
        for a in absences:
            base = self.POSITION_BASE_IMPACT.get(a.position.upper(), 0.05)
            importance = float(np.clip(a.importance, 0.0, 1.0))
            # Bayesian update: spadek p_win o (3% do 12%) skalowany przez importance
            single_impact = base * (0.5 + 1.5 * importance)
            single_impact = float(np.clip(single_impact, 0.0, self.MAX_TOTAL_IMPACT))
            total += single_impact
            details.append({
                "player": a.player_name,
                "position": a.position,
                "importance": importance,
                "reason": a.reason,
                "impact": round(single_impact, 4),
            })

        # Diminishing returns — wielu nieobecnych nie sumuje się liniowo do nieskończoności
        total_clamped = float(np.clip(total, 0.0, self.MAX_TOTAL_IMPACT))
        # Efekt nasycenia (im więcej absencji, tym mniejszy marginalny wpływ kolejnej)
        saturated = self.MAX_TOTAL_IMPACT * (1 - np.exp(-total_clamped / self.MAX_TOTAL_IMPACT * 2))

        return {
            "impact_score": round(-float(np.clip(saturated, 0.0, self.MAX_TOTAL_IMPACT)), 4),
            "details": details,
            "n_absences": len(absences),
        }

    def apply_to_pwin(self, p_win: float, impact_home: float, impact_away: float) -> Dict:
        """
        Aplikuje impacty obu drużyn do p_win.
        impact_home/away są ujemne (np. -0.05).
        Pozytywny impact rywala (jego absencje) zwiększa p_win drużyny gospodarza.
        """
        # Absencje gospodarza obniżają jego p_win, absencje gościa zwiększają
        adjustment = impact_home - impact_away  # obie ujemne, więc np. (-0.05) - (-0.08) = +0.03
        p_adjusted = float(np.clip(p_win + adjustment, 0.01, 0.99))
        return {
            "p_win_before_injury": round(float(p_win), 4),
            "p_win_after_injury": round(p_adjusted, 4),
            "adjustment": round(adjustment, 4),
            "impact_home": round(impact_home, 4),
            "impact_away": round(impact_away, 4),
        }


# =============================================================================
# MODULE 4: WEATHER & PITCH IMPACT MODULE
# =============================================================================

@dataclass
class WeatherConditions:
    condition: str = "clear"       # "clear", "rain", "heavy_rain", "snow", "wind", "fog"
    temperature_c: float = 18.0
    wind_speed_kmh: float = 5.0
    humidity_pct: float = 50.0
    pitch_type: str = "natural"    # "natural", "artificial", "hybrid"


class WeatherPitchImpactModule:
    """
    Modeluje wpływ warunków atmosferycznych i typu murawy na rynki goli
    i remisów.

    Reguły bazowe:
      - Deszcz → spadek liczby goli (~15%), wzrost p(draw)
      - Wysoka temperatura (>28C) → wzrost zmęczenia w 2. połowie
      - Sztuczna murawa → przewaga dla drużyn przyzwyczajonych (home team bias)
      - Wiatr → spadek precyzji strzałów, lekki spadek goli
    """

    def __init__(self):
        pass

    def goals_multiplier(self, weather: WeatherConditions) -> float:
        """Mnożnik dla oczekiwanej liczby goli (mu w Dixon-Coles / Poisson)."""
        mult = 1.0

        if weather.condition in ("rain",):
            mult *= 0.93   # ~ -7% (umiarkowany deszcz)
        elif weather.condition in ("heavy_rain", "snow"):
            mult *= 0.85   # ~ -15%
        elif weather.condition == "wind" or weather.wind_speed_kmh > 30:
            mult *= 0.95

        if weather.temperature_c > 28:
            mult *= 0.97   # spadek tempa w 2. połowie przez upał

        if weather.temperature_c < 0:
            mult *= 0.96   # mróz — twardsza murawa, mniej płynna gra

        return float(np.clip(mult, 0.7, 1.05))

    def draw_probability_adjustment(self, weather: WeatherConditions) -> float:
        """Addytywna korekta dla p_draw."""
        adj = 0.0
        if weather.condition in ("rain",):
            adj += 0.02
        elif weather.condition in ("heavy_rain", "snow"):
            adj += 0.04
        if weather.wind_speed_kmh > 30:
            adj += 0.015
        return float(np.clip(adj, 0.0, 0.06))

    def home_advantage_adjustment(self, weather: WeatherConditions,
                                   home_team_artificial_familiar: bool = False) -> float:
        """
        Korekta home_advantage z tytułu sztucznej murawy — jeśli gospodarz
        gra regularnie na sztucznej murawie, ma przewagę gdy mecz odbywa
        się na takiej murawie (gość nieprzyzwyczajony).
        """
        adj = 0.0
        if weather.pitch_type in ("artificial", "hybrid") and home_team_artificial_familiar:
            adj += 0.05
        return float(np.clip(adj, 0.0, 0.08))

    def second_half_fatigue_bonus(self, weather: WeatherConditions) -> float:
        """
        Dodatkowy bonus do 'fatigue' dimension reprezentujący kumulacyjny
        wpływ wysokiej temperatury na drugą połowę meczu.
        """
        if weather.temperature_c > 30:
            return 0.10
        if weather.temperature_c > 28:
            return 0.06
        return 0.0

    def apply_to_match_params(self, match_params: np.ndarray, dimensions: List[str],
                               weather: WeatherConditions,
                               home_team_artificial_familiar: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Modyfikuje wektor match_params w miejscach 'fatigue' i 'home_advantage'
        (jeśli istnieją w dimensions) na podstawie pogody. Zwraca
        zmodyfikowany wektor + raport korekt (do logowania).
        """
        params = match_params.copy()
        report = {
            "goals_multiplier": round(self.goals_multiplier(weather), 4),
            "draw_adjustment": round(self.draw_probability_adjustment(weather), 4),
            "home_adv_adjustment": round(
                self.home_advantage_adjustment(weather, home_team_artificial_familiar), 4
            ),
            "fatigue_bonus": round(self.second_half_fatigue_bonus(weather), 4),
            "condition": weather.condition,
            "pitch_type": weather.pitch_type,
        }

        if "fatigue" in dimensions:
            idx = dimensions.index("fatigue")
            params[idx] = float(np.clip(params[idx] + report["fatigue_bonus"], 0.0, 1.0))

        if "home_advantage" in dimensions:
            idx = dimensions.index("home_advantage")
            params[idx] = float(np.clip(params[idx] + report["home_adv_adjustment"], 0.0, 1.0))

        return params, report


# =============================================================================
# MODULE 5: PSEUDO-LABELING + SELF-TRAINING
# =============================================================================

class PseudoLabelingEngine:
    """
    Semi-supervised self-training: dla meczów bez znanego wyniku (tylko
    statystyki przedmeczowe), generuje pseudo-etykiety na podstawie
    aktualnego modelu, filtruje przez próg pewności (confidence threshold),
    i zwraca rozszerzony zbiór treningowy (X_aug, y_aug).

    Implementuje klasyczny self-training loop używany do "treningu na
    milionach meczów bez wyniku".
    """

    def __init__(self, confidence_threshold: float = 0.85, max_pseudo_ratio: float = 0.5):
        """
        confidence_threshold: tylko predykcje |p-0.5|*2 >= threshold są
            traktowane jako wystarczająco pewne, aby użyć jako pseudo-label.
        max_pseudo_ratio: maksymalny rozmiar zbioru pseudo-etykiet względem
            oryginalnego zbioru treningowego (zapobiega dominacji pseudo-danych).
        """
        self.confidence_threshold = confidence_threshold
        self.max_pseudo_ratio = max_pseudo_ratio
        self.last_stats: Dict = {}

    def generate_pseudo_labels(self, predict_fn: Callable[[np.ndarray], np.ndarray],
                                X_unlabeled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        predict_fn: funkcja X -> p_win array (np. tree_layer.predict)
        X_unlabeled: cechy meczów bez wyniku, shape (N, n_features)

        Zwraca: (X_pseudo, y_pseudo, stats)
        """
        if X_unlabeled is None or len(X_unlabeled) == 0:
            return np.empty((0, 0)), np.empty((0,)), {"n_selected": 0, "n_total": 0}

        preds = predict_fn(X_unlabeled)
        preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)

        confidence = np.abs(preds - 0.5) * 2.0  # 0 = niepewny (0.5), 1 = bardzo pewny (0 lub 1)
        mask = confidence >= self.confidence_threshold

        X_sel = X_unlabeled[mask]
        y_sel = preds[mask]

        stats = {
            "n_total": int(len(X_unlabeled)),
            "n_selected": int(mask.sum()),
            "selection_rate": round(float(mask.mean()), 4) if len(mask) else 0.0,
            "mean_confidence_selected": round(float(confidence[mask].mean()), 4) if mask.sum() > 0 else None,
        }
        self.last_stats = stats
        return X_sel, y_sel, stats

    def augment_training_set(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_pseudo: np.ndarray, y_pseudo: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Łączy oryginalny zbiór treningowy z pseudo-etykietowanym, respektując
        max_pseudo_ratio.
        """
        if X_pseudo.size == 0:
            return X_train, y_train, {"n_pseudo_used": 0}

        max_pseudo_n = int(len(X_train) * self.max_pseudo_ratio)
        n_pseudo_used = min(len(X_pseudo), max_pseudo_n)

        if n_pseudo_used < len(X_pseudo):
            idx = np.random.RandomState(42).choice(len(X_pseudo), n_pseudo_used, replace=False)
            X_pseudo = X_pseudo[idx]
            y_pseudo = y_pseudo[idx]

        X_aug = np.vstack([X_train, X_pseudo]) if n_pseudo_used > 0 else X_train
        y_aug = np.concatenate([y_train, y_pseudo]) if n_pseudo_used > 0 else y_train

        return X_aug, y_aug, {"n_pseudo_used": int(n_pseudo_used), "n_original": int(len(X_train))}


# =============================================================================
# MODULE 6: CONTRASTIVE LEARNING — TEAM EMBEDDINGS
# =============================================================================

class TeamEmbeddingContrastive:
    """
    Lekkie embeddingi drużyn uczone metodą contrastive learning (InfoNCE-style,
    czysty numpy). Drużyny o podobnym stylu gry / sile zbliżają się w przestrzeni
    embeddingów; różne — odpychają się.

    Zastosowanie:
      - similarity(team_a, team_b) — jak bardzo podobne style gry
      - transfer learning między ligami: drużyna z Championship podobna do
        drużyny Premier League → można "transferować" jej cechy/oceny.
    """

    def __init__(self, embedding_dim: int = 8, lr: float = 0.02, temperature: float = 0.1):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.temperature = temperature
        self.embeddings: Dict[str, np.ndarray] = {}
        self._rng = np.random.RandomState(123)

    def _get_or_init(self, team: str) -> np.ndarray:
        if team not in self.embeddings:
            self.embeddings[team] = self._rng.normal(0, 0.5, self.embedding_dim)
        return self.embeddings[team]

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v) + 1e-9
        return v / norm

    def similarity(self, team_a: str, team_b: str) -> float:
        """Cosine similarity między embeddingami dwóch drużyn, w [-1, 1]."""
        ea = self._normalize(self._get_or_init(team_a))
        eb = self._normalize(self._get_or_init(team_b))
        return float(np.clip(np.dot(ea, eb), -1.0, 1.0))

    def contrastive_update(self, anchor: str, positive: str, negatives: List[str]):
        """
        Krok aktualizacji InfoNCE: 'positive' powinien być bardziej podobny
        do 'anchor' niż wszystkie 'negatives' (np. drużyny o podobnym stylu
        vs przypadkowe drużyny z innej ligi).
        """
        e_anchor = self._get_or_init(anchor)
        e_pos = self._get_or_init(positive)
        e_negs = [self._get_or_init(n) for n in negatives] if negatives else []

        a_norm = self._normalize(e_anchor)
        p_norm = self._normalize(e_pos)

        sim_pos = np.dot(a_norm, p_norm) / self.temperature
        sims_neg = [np.dot(a_norm, self._normalize(n)) / self.temperature for n in e_negs]

        all_sims = np.array([sim_pos] + sims_neg)
        # softmax probability that 'positive' is the matching pair
        probs = np.exp(all_sims - np.max(all_sims))
        probs /= probs.sum() + 1e-12
        prob_pos = probs[0]

        # Gradient ascent w stronę zbliżenia anchor-positive (proporcjonalnie do (1 - prob_pos))
        grad = (1.0 - prob_pos) * self.lr
        self.embeddings[anchor] = e_anchor + grad * (e_pos - e_anchor)
        self.embeddings[positive] = e_pos + grad * (e_anchor - e_pos)

        # Odpychanie od negatywów
        for i, neg in enumerate(negatives):
            e_neg = self.embeddings[neg]
            push = probs[i + 1] * self.lr
            self.embeddings[neg] = e_neg - push * (e_anchor - e_neg)

        return {"prob_pos": round(float(prob_pos), 4), "n_negatives": len(negatives)}

    def nearest_teams(self, team: str, candidates: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """Zwraca top_n najbardziej podobnych drużyn z listy candidates."""
        sims = [(c, self.similarity(team, c)) for c in candidates if c != team]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_n]

    def transfer_feature_adjustment(self, target_team: str, source_team: str,
                                     source_feature_value: float) -> Dict:
        """
        Transfer learning: jeśli target_team (np. z Championship) jest
        podobna do source_team (np. z Premier League), skaluje
        source_feature_value przez similarity, jako "transferowaną" wartość
        cechy dla target_team.
        """
        sim = self.similarity(target_team, source_team)
        sim_clipped = float(np.clip(sim, 0.0, 1.0))
        transferred_value = source_feature_value * sim_clipped + 0.5 * (1 - sim_clipped)
        return {
            "similarity": round(sim, 4),
            "transferred_value": round(float(np.clip(transferred_value, 0.0, 1.0)), 4),
        }


# =============================================================================
# MODULE 7: DYNAMIC THRESHOLD TUNING (adaptacyjny próg BANEBET)
# =============================================================================

class DynamicThresholdTuner:
    """
    Zamiast stałej drabinki progów BANEBET v5.4, dostarcza modyfikator
    progu (delta) zależny od:
      - ligi (np. La Liga ma wyższą predykcyjność niż Ligue 1)
      - typu meczu (derby = wysoka zmienność -> wyższy próg)
      - godziny/dnia tygodnia (mecze w tygodniu po europejskich pucharach
        -> więcej rotacji -> wyższa niepewność -> wyższy próg)

    Output jest deltą dodawaną do base_threshold z evaluate_banebet_v5_4_decision,
    a finalny próg jest przycinany do [0.05, 0.95].
    """

    # Im wyższa wartość, tym wyższa "predykcyjność" ligi (niższy wymagany próg)
    LEAGUE_PREDICTABILITY: Dict[str, float] = {
        "la_liga": 0.04,
        "premier_league": 0.02,
        "bundesliga": 0.03,
        "serie_a": 0.01,
        "ligue_1": -0.02,
        "eredivisie": 0.00,
        "ekstraklasa": -0.01,
        "championship": -0.015,
        "default": 0.0,
    }

    DAY_OF_WEEK_ADJUST: Dict[int, float] = {
        # 1=Sunday ... 7=Saturday (zgodnie z konwencją event_create_v0 1-indexed)
        2: 0.015,  # Monday — po weekendowych meczach, rotacje
        3: 0.01,   # Tuesday — Champions League/Europa midweek
        4: 0.01,   # Wednesday — midweek
        5: -0.005, # Thursday — Europa League, ale mniej rotacji
    }

    def __init__(self):
        pass

    def threshold_delta(self, league: str = "default",
                         is_derby: bool = False,
                         day_of_week: Optional[int] = None,
                         high_stakes: bool = False) -> Dict:
        """
        Zwraca deltę progu (do dodania do base_threshold) oraz rozbicie
        czynników.
        """
        delta = 0.0
        breakdown = {}

        league_factor = -self.LEAGUE_PREDICTABILITY.get(league, self.LEAGUE_PREDICTABILITY["default"])
        delta += league_factor
        breakdown["league_factor"] = round(league_factor, 4)

        derby_factor = 0.05 if is_derby else 0.0
        delta += derby_factor
        breakdown["derby_factor"] = round(derby_factor, 4)

        dow_factor = 0.0
        if day_of_week is not None:
            dow_factor = self.DAY_OF_WEEK_ADJUST.get(day_of_week, 0.0)
        delta += dow_factor
        breakdown["day_of_week_factor"] = round(dow_factor, 4)

        stakes_factor = -0.03 if high_stakes else 0.0  # mecz "must-win" — drużyny grają bardziej zachowawczo i przewidywalnie
        delta += stakes_factor
        breakdown["high_stakes_factor"] = round(stakes_factor, 4)

        return {"delta": round(float(delta), 4), "breakdown": breakdown}

    def apply_to_decision(self, base_threshold: float, mean_p: float,
                           league: str = "default", is_derby: bool = False,
                           day_of_week: Optional[int] = None,
                           high_stakes: bool = False) -> Dict:
        """
        Stosuje dynamiczną deltę do base_threshold i zwraca informację,
        czy mean_p przekracza nowy próg.
        """
        delta_info = self.threshold_delta(league, is_derby, day_of_week, high_stakes)
        adjusted_threshold = float(np.clip(base_threshold + delta_info["delta"], 0.05, 0.95))
        passes = mean_p >= adjusted_threshold
        return {
            "base_threshold": round(base_threshold, 4),
            "adjusted_threshold": round(adjusted_threshold, 4),
            "delta": delta_info["delta"],
            "breakdown": delta_info["breakdown"],
            "passes_dynamic_threshold": bool(passes),
        }


# =============================================================================
# MODULE 8: GPU ACCELERATION + BATCH INFERENCE (config helper)
# =============================================================================

@dataclass
class GPUBatchConfig:
    """
    Konfiguracja dla treningu/inferencji na GPU (XGBoost/CatBoost/LightGBM).
    Nie wymusza GPU jeśli niedostępne — generuje parametry, które są
    bezpiecznie ignorowane przez biblioteki, jeśli GPU brak.

    Użycie:
        cfg = GPUBatchConfig(use_gpu=True, batch_size=8192)
        xgb_params.update(cfg.xgb_gpu_params())
    """
    use_gpu: bool = False
    batch_size: int = 4096
    gpu_id: int = 0
    max_bin: int = 256  # wpływa na pamięć GPU dla histogram-based tree methods

    def xgb_gpu_params(self) -> Dict:
        if not self.use_gpu:
            return {"tree_method": "hist", "device": "cpu"}
        return {
            "tree_method": "hist",
            "device": f"cuda:{self.gpu_id}",
            "max_bin": self.max_bin,
        }

    def lgb_gpu_params(self) -> Dict:
        if not self.use_gpu:
            return {"device_type": "cpu"}
        return {
            "device_type": "gpu",
            "gpu_device_id": self.gpu_id,
            "max_bin": self.max_bin,
        }

    def cb_gpu_params(self) -> Dict:
        if not self.use_gpu:
            return {"task_type": "CPU"}
        return {
            "task_type": "GPU",
            "devices": str(self.gpu_id),
        }

    def batch_iter(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Generator batchy dla treningu/predykcji na dużych zbiorach
        (100k+ meczów), zapobiega OOM nawet bez GPU.
        """
        n = len(X)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            if y is not None:
                yield X[start:end], y[start:end]
            else:
                yield X[start:end]

    def recommended_for_dataset(self, n_samples: int) -> Dict:
        """
        Heurystyka: dla dużych zbiorów (100k+) sugeruje GPU + większy batch.
        Hardware referencyjny: Lenovo Legion Slim 5, RTX 4050 6GB.
        """
        if n_samples >= 100_000:
            return {
                "recommendation": "GPU + batched training (RTX 4050 6GB: max_bin<=128, batch_size<=4096)",
                "use_gpu": True,
                "suggested_batch_size": 4096,
                "suggested_max_bin": 128,
            }
        elif n_samples >= 20_000:
            return {
                "recommendation": "GPU opcjonalnie, CPU hist również wystarczający",
                "use_gpu": True,
                "suggested_batch_size": 8192,
                "suggested_max_bin": 256,
            }
        else:
            return {
                "recommendation": "CPU wystarczający dla tego rozmiaru danych",
                "use_gpu": False,
                "suggested_batch_size": len(range(0, n_samples, self.batch_size)) and n_samples,
                "suggested_max_bin": 256,
            }


# =============================================================================
# MODULE 9: MULTI-LOSS OPTIMIZATION
# =============================================================================

class MultiLossEvaluator:
    """
    Łączona ocena predykcji przez 3 metryki jednocześnie:
      - MAE (jak dotychczas w v4.0)
      - Ranking loss (Spearman-style: czy kolejność predykcji odpowiada
        kolejności rzeczywistych wyników)
      - Calibration loss (ECE — Expected Calibration Error)
      - AUC (dyskryminacja binarna win/no-win)

    Nie zastępuje treningu modeli (które wciąż optymalizują MAE w v4.0),
    ale dostarcza dodatkowy multi-metric score używany do:
      - selekcji hiperparametrów (jako tie-breaker dla Optuna)
      - monitorowania jakości modelu w czasie (drift detection)
    """

    def __init__(self, n_calibration_bins: int = 10,
                 weight_mae: float = 0.4, weight_rank: float = 0.25,
                 weight_calib: float = 0.2, weight_auc: float = 0.15):
        self.n_bins = n_calibration_bins
        self.w_mae = weight_mae
        self.w_rank = weight_rank
        self.w_calib = weight_calib
        self.w_auc = weight_auc

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    def ranking_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        1 - Spearman correlation (im niższy, tym lepsza zgodność kolejności).
        Zwraca wartość w [0, 2] (0 = perfekcyjna zgodność rankingu).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) < 2:
            return 0.0

        def rankdata(a):
            order = np.argsort(a)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(a))
            return ranks

        rt = rankdata(y_true)
        rp = rankdata(y_pred)
        rt_c = rt - rt.mean()
        rp_c = rp - rp.mean()
        denom = (np.sqrt((rt_c**2).sum()) * np.sqrt((rp_c**2).sum())) + 1e-9
        spearman = float(np.dot(rt_c, rp_c) / denom)
        return float(np.clip(1.0 - spearman, 0.0, 2.0))

    def calibration_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Expected Calibration Error (ECE): średnia różnica między
        przewidywanym p i empiryczną częstością w binach prawdopodobieństwa.
        y_true traktowane jako binarne (>0.5 -> 1).
        """
        y_true_bin = (np.asarray(y_true) > 0.5).astype(float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)

        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(y_pred)
        if n == 0:
            return 0.0

        for i in range(self.n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (y_pred >= lo) & (y_pred < hi if i < self.n_bins - 1 else y_pred <= hi)
            if mask.sum() == 0:
                continue
            bin_conf = float(y_pred[mask].mean())
            bin_acc = float(y_true_bin[mask].mean())
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

        return float(ece)

    def auc_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        y_true_bin = (np.asarray(y_true) > 0.5).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            return None
        if SKLEARN_METRICS_AVAILABLE:
            try:
                return float(roc_auc_score(y_true_bin, y_pred))
            except Exception:
                pass
        # Fallback: ręczny AUC (Mann-Whitney U)
        pos = np.asarray(y_pred)[y_true_bin == 1]
        neg = np.asarray(y_pred)[y_true_bin == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        comparisons = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        return float(comparisons / (len(pos) * len(neg)))

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        mae = self.mae(y_true, y_pred)
        rank = self.ranking_loss(y_true, y_pred)
        calib = self.calibration_error(y_true, y_pred)
        auc = self.auc_score(y_true, y_pred)

        # Composite score — niższy = lepszy (wszystkie składniki są "im mniej tym lepiej",
        # AUC odwracamy bo wyższy AUC = lepszy)
        auc_loss = (1.0 - auc) if auc is not None else 0.5
        composite = (self.w_mae * mae + self.w_rank * rank +
                      self.w_calib * calib + self.w_auc * auc_loss)

        return {
            "mae": round(mae, 5),
            "ranking_loss": round(rank, 5),
            "calibration_error": round(calib, 5),
            "auc": round(auc, 5) if auc is not None else None,
            "composite_loss": round(float(composite), 5),
        }
# =============================================================================
# =============================================================================
#   HYBRID BETTING ENGINE v1.0 — GŁÓWNY SILNIK ROZSZERZONY
#
#   Dziedziczy po UniversalBettingEngineV3 (v4.0) — WSZYSTKO z v4.0
#   pozostaje aktywne (tree_layer, mebn, dixon_coles, elo, adaptive,
#   calibrator, BANEBET v5.4, SHAP, Optuna, itd.).
#
#   Dodaje 9 nowych modułów jako kompozycję (composition), wołanych
#   w nadpisanej metodzie predict() -> super().predict() + warstwy hybrydowe.
# =============================================================================
# =============================================================================

@dataclass
class HybridMatchContext:
    """
    Opcjonalny kontekst meczu — kontener dla wszystkich nowych danych
    wejściowych potrzebnych modułom hybrydowym. Wszystkie pola opcjonalne,
    domyślnie neutralne (brak wpływu na predykcję v4.0).
    """
    # Module 1 — Market-Based Calibration
    odds_1: List[float] = field(default_factory=list)
    odds_x: List[float] = field(default_factory=list)
    odds_2: List[float] = field(default_factory=list)

    # Module 2 — Temporal Attention
    home_recent_form: Optional[np.ndarray] = None  # wektor (feature_dim,) ostatniego meczu (do dopisania do historii)
    away_recent_form: Optional[np.ndarray] = None

    # Module 3 — Injuries/Suspensions
    home_absences: List[PlayerAbsence] = field(default_factory=list)
    away_absences: List[PlayerAbsence] = field(default_factory=list)

    # Module 4 — Weather
    weather: Optional[WeatherConditions] = None
    home_team_artificial_familiar: bool = False

    # Module 6 — Contrastive embeddings (transfer learning)
    transfer_source_team: Optional[str] = None
    transfer_source_feature_value: Optional[float] = None
    transfer_target_feature_idx: Optional[int] = None

    # Module 7 — Dynamic Threshold
    league: str = "default"
    is_derby: bool = False
    day_of_week: Optional[int] = None
    high_stakes: bool = False

    # Module 5 — Pseudo-labeling (batch, niezależne od pojedynczego meczu)
    # obsługiwane przez metody engine, nie przez predict()


class HybridBettingEngineV1(UniversalBettingEngineV3):
    """
    HYBRID BETTING MODEL v1.0 — BANEBET PRO

    Potwór łączący wszystkie etapy wiedzy v4.0 (TIER 1-5: ensemble 9 modeli,
    stacking, Optuna, SHAP, Dixon-Coles, dynamic Elo, kalibracja, BANEBET v5.4,
    adaptive learning) z 9 nowymi modułami hybrydowymi (TIER 6):

      TIER 6.1: Market-Based Calibration (MBC)
      TIER 6.2: Temporal Attention Mechanism (forma w czasie)
      TIER 6.3: Injury/Suspension Impact Model
      TIER 6.4: Weather & Pitch Impact Module
      TIER 6.5: Pseudo-labeling + Self-training
      TIER 6.6: Contrastive Learning team embeddings
      TIER 6.7: Dynamic Threshold Tuning
      TIER 6.8: GPU Acceleration + Batch Inference config
      TIER 6.9: Multi-Loss Optimization (monitoring)

    NIC z v4.0 NIE ZOSTAŁO USUNIĘTE — wszystkie metody, pola i zachowania
    UniversalBettingEngineV3 są dostępne i aktywne. Hybrid dodaje nową
    warstwę na wierzchu.
    """

    def __init__(self, sport: str, use_optuna: bool = True,
                 optuna_trials: int = 30,
                 calibration_method: str = "isotonic",
                 # --- nowe parametry TIER 6 ---
                 mbc_blend_weight: float = 0.35,
                 temporal_window: int = 20,
                 temporal_feature_dim: int = 5,
                 gpu_config: Optional[GPUBatchConfig] = None,
                 pseudo_confidence_threshold: float = 0.85,
                 pseudo_max_ratio: float = 0.5,
                 contrastive_embedding_dim: int = 8):
        super().__init__(sport, use_optuna=use_optuna,
                          optuna_trials=optuna_trials,
                          calibration_method=calibration_method)

        # TIER 6.1 — Market-Based Calibration
        self.mbc = MarketBasedCalibrator(blend_weight=mbc_blend_weight)

        # TIER 6.2 — Temporal Attention
        self.temporal_attention = TemporalAttentionForm(
            window=temporal_window, feature_dim=temporal_feature_dim
        )

        # TIER 6.3 — Injury/Suspension
        self.injury_model = InjurySuspensionImpactModel()

        # TIER 6.4 — Weather & Pitch
        self.weather_module = WeatherPitchImpactModule()

        # TIER 6.5 — Pseudo-labeling
        self.pseudo_labeler = PseudoLabelingEngine(
            confidence_threshold=pseudo_confidence_threshold,
            max_pseudo_ratio=pseudo_max_ratio,
        )

        # TIER 6.6 — Contrastive team embeddings
        self.team_embeddings = TeamEmbeddingContrastive(embedding_dim=contrastive_embedding_dim)

        # TIER 6.7 — Dynamic Threshold Tuning
        self.threshold_tuner = DynamicThresholdTuner()

        # TIER 6.8 — GPU/Batch config
        self.gpu_config = gpu_config or GPUBatchConfig(use_gpu=False)

        # TIER 6.9 — Multi-Loss monitoring
        self.multi_loss = MultiLossEvaluator()
        self._last_multi_loss: Optional[Dict] = None

        print(f"[HYBRID v1.0] Inicjalizacja TIER 6 zakończona dla: {self.config.name}")
        print(f"[HYBRID v1.0] Moduły: MBC | TemporalAttention | Injury/Suspension | "
              f"Weather/Pitch | PseudoLabel | ContrastiveEmbed | DynThreshold | "
              f"GPU/Batch | MultiLoss")

    # ------------------------------------------------------------------
    # TRENING — TIER 6.8: opcjonalny self-training z pseudo-labelami
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
              X_unlabeled: Optional[np.ndarray] = None,
              use_gpu_config: bool = True) -> Dict:
        """
        Trening rozszerzony:
          1. (opcjonalnie) pseudo-labeling na X_unlabeled przed treningiem
             finalnym — wymaga wstępnie wytrenowanego modelu, więc robimy
             dwuetapowo: trening bazowy -> generacja pseudo-labeli -> trening
             finalny na rozszerzonym zbiorze.
          2. Standardowy trening v4.0 (TreeLayerV3, MEBN, AdaptiveLearner).
          3. TIER 6.9: ocena multi-loss na danych treningowych (monitoring).

        Zwraca dict z informacjami o treningu (oprócz standardowego print()
        z v4.0).
        """
        report: Dict = {"pseudo_labeling": None, "multi_loss": None,
                         "gpu_recommendation": None}

        # --- GPU/batch recommendation (TIER 6.8) ---
        if use_gpu_config:
            report["gpu_recommendation"] = self.gpu_config.recommended_for_dataset(len(X))

        X_final, y_final = X, y

        # --- TIER 6.5: Pseudo-labeling (jeśli dostarczono dane nieoznakowane) ---
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            print("[HYBRID] TIER 6.5 — Trening bazowy przed pseudo-labelingiem...")
            super().train(X, y, n_splits=n_splits)

            print("[HYBRID] TIER 6.5 — Generowanie pseudo-etykiet...")
            X_pseudo, y_pseudo, pl_stats = self.pseudo_labeler.generate_pseudo_labels(
                predict_fn=self.tree_layer.predict, X_unlabeled=X_unlabeled
            )
            X_final, y_final, aug_stats = self.pseudo_labeler.augment_training_set(
                X, y, X_pseudo, y_pseudo
            )
            report["pseudo_labeling"] = {**pl_stats, **aug_stats}
            print(f"[HYBRID] TIER 6.5 — Pseudo-labeling: {report['pseudo_labeling']}")

        # --- Standardowy trening v4.0 (TIER 1-5) na (rozszerzonym) zbiorze ---
        print("[HYBRID] TIER 1-5 — Trening ensemble v4.0 (zachowany w pełni)...")
        super().train(X_final, y_final, n_splits=n_splits)

        # --- TIER 6.9: Multi-loss monitoring na danych treningowych ---
        try:
            y_pred_train = self.tree_layer.predict(X_final)
            ml = self.multi_loss.evaluate(y_final, y_pred_train)
            self._last_multi_loss = ml
            report["multi_loss"] = ml
            print(f"[HYBRID] TIER 6.9 — Multi-loss (train): {ml}")
        except Exception as e:
            warnings.warn(f"[HYBRID] TIER 6.9 multi-loss błąd: {e}")

        print("[HYBRID v1.0] Trening zakończony — wszystkie tiery aktywne.")
        return report

    # ------------------------------------------------------------------
    # PREDYKCJA HYBRYDOWA
    # ------------------------------------------------------------------

    def predict(self, match_params: np.ndarray,
                team_home: Optional[str] = None,
                team_away: Optional[str] = None,
                context: Optional[HybridMatchContext] = None) -> Dict:
        """
        Predykcja hybrydowa:
          1. Modyfikacja match_params PRZED predykcją bazową:
             - TIER 6.4: Weather/Pitch (modyfikuje fatigue, home_advantage)
             - TIER 6.6: Contrastive transfer (opcjonalna modyfikacja
               wybranego wymiaru przez transfer z podobnej drużyny)
          2. Wywołanie super().predict() — CAŁY pipeline v4.0 (TIER 1-5)
             działa bez zmian (ensemble, MEBN, Dixon-Coles, Elo, kalibracja,
             SHAP, BANEBET v5.4).
          3. Modyfikacja wyniku PO predykcji bazowej:
             - TIER 6.2: Temporal Attention — attended_form_score per drużyna
             - TIER 6.3: Injury/Suspension impact na p_win
             - TIER 6.1: Market-Based Calibration — blend p_win z rynkiem
             - TIER 6.7: Dynamic Threshold — nowy próg dla BANEBET v5.4
          4. Złożony wynik finalny "hybrid_v1" w results.
        """
        if context is None:
            context = HybridMatchContext()

        params = match_params.copy()
        hybrid_report: Dict = {}

        # --- TIER 6.4: Weather & Pitch (PRZED predykcją bazową) ---
        if context.weather is not None:
            params, weather_report = self.weather_module.apply_to_match_params(
                params, self.config.dimensions, context.weather,
                home_team_artificial_familiar=context.home_team_artificial_familiar,
            )
            hybrid_report["weather"] = weather_report
        else:
            hybrid_report["weather"] = None

        # --- TIER 6.6: Contrastive transfer (PRZED predykcją bazową) ---
        if (context.transfer_source_team and context.transfer_source_feature_value is not None
                and context.transfer_target_feature_idx is not None
                and team_home):
            transfer = self.team_embeddings.transfer_feature_adjustment(
                target_team=team_home,
                source_team=context.transfer_source_team,
                source_feature_value=context.transfer_source_feature_value,
            )
            idx = context.transfer_target_feature_idx
            if 0 <= idx < len(params):
                params[idx] = transfer["transferred_value"]
            hybrid_report["contrastive_transfer"] = transfer
        else:
            hybrid_report["contrastive_transfer"] = None

        # ==================================================================
        # CAŁY PIPELINE v4.0 (TIER 1-5) — NIENARUSZONY
        # ==================================================================
        results = super().predict(params, team_home=team_home, team_away=team_away)

        # --- TIER 6.2: Temporal Attention (PO predykcji bazowej) ---
        temporal_info = {}
        if team_home:
            if context.home_recent_form is not None:
                self.temporal_attention.add_match(team_home, context.home_recent_form)
            temporal_info["home"] = self.temporal_attention.attended_form(team_home, params)
        if team_away:
            if context.away_recent_form is not None:
                self.temporal_attention.add_match(team_away, context.away_recent_form)
            temporal_info["away"] = self.temporal_attention.attended_form(team_away, params)
        hybrid_report["temporal_attention"] = temporal_info

        # --- TIER 6.3: Injury/Suspension impact ---
        injury_home = self.injury_model.compute_impact(context.home_absences)
        injury_away = self.injury_model.compute_impact(context.away_absences)
        injury_application = self.injury_model.apply_to_pwin(
            results["p_win"],
            injury_home["impact_score"],
            injury_away["impact_score"],
        )
        hybrid_report["injury_suspension"] = {
            "home": injury_home,
            "away": injury_away,
            "application": injury_application,
        }

        # p_win po korekcie injury — używany jako wejście do MBC i finalnych rynków
        p_win_post_injury = injury_application["p_win_after_injury"]

        # --- TIER 6.1: Market-Based Calibration ---
        mbc_result = None
        if context.odds_1 and context.odds_x and context.odds_2:
            market_probs = self.mbc.remove_overround(context.odds_1, context.odds_x, context.odds_2)
            mbc_result = self.mbc.calibrate_with_market(p_win_post_injury, market_probs, outcome_key="p1")
            self.mbc.record(mbc_result)
        hybrid_report["market_based_calibration"] = mbc_result

        # Finalne p_win hybrydowe: MBC blend (jeśli dostępny) inaczej p_win_post_injury
        p_win_final = mbc_result["p_blended"] if mbc_result else p_win_post_injury

        # Przelicz rynki pochodne na nowym p_win_final (zachowując strukturę v4.0)
        p_draw = float(results.get("p_draw", 0.0))
        p_loss = max(0.01, 1.0 - p_win_final - p_draw)
        total = p_win_final + p_draw + p_loss
        p_win_final /= total
        p_draw_norm = p_draw / total
        p_loss_norm = p_loss / total

        hybrid_report["final_probabilities"] = {
            "p_win": round(p_win_final, 4),
            "p_draw": round(p_draw_norm, 4),
            "p_loss": round(p_loss_norm, 4),
            "fair_1": round(1.0 / max(p_win_final, 1e-6), 3),
            "fair_x": round(1.0 / max(p_draw_norm, 1e-6), 3) if p_draw_norm > 0.005 else None,
            "fair_2": round(1.0 / max(p_loss_norm, 1e-6), 3),
        }

        # --- TIER 6.7: Dynamic Threshold Tuning ---
        banebet_v54 = results.get("banebet_v54", {})
        base_threshold = banebet_v54.get("metadata", {}).get("ladder_threshold", 0.5)
        dyn_threshold = self.threshold_tuner.apply_to_decision(
            base_threshold=base_threshold,
            mean_p=banebet_v54.get("mean_p", p_win_final),
            league=context.league,
            is_derby=context.is_derby,
            day_of_week=context.day_of_week,
            high_stakes=context.high_stakes,
        )
        hybrid_report["dynamic_threshold"] = dyn_threshold

        # --- Decyzja hybrydowa finalna (TIER 6 overlay na BANEBET v5.4) ---
        original_action = banebet_v54.get("action", "NO_BET")
        hybrid_action = original_action
        if original_action == "BET" and not dyn_threshold["passes_dynamic_threshold"]:
            hybrid_action = "NO_BET"
        elif original_action == "NO_BET" and dyn_threshold["passes_dynamic_threshold"] \
                and banebet_v54.get("mean_p", 0) >= dyn_threshold["adjusted_threshold"]:
            # Tylko jeśli oryginalny mean_p przekracza nowy (niższy) próg
            hybrid_action = "BET"

        # Value rating z MBC może też zawetować zakład
        if mbc_result and mbc_result["value_rating"] == "AVOID_TRAP":
            hybrid_action = "NO_BET"

        hybrid_report["hybrid_decision"] = {
            "original_action": original_action,
            "hybrid_action": hybrid_action,
            "reason": "dynamic_threshold + injury + MBC overlay",
        }

        # --- TIER 6.9: Multi-loss info (ostatni znany stan z treningu) ---
        hybrid_report["multi_loss_last_train"] = self._last_multi_loss

        results["hybrid_v1"] = hybrid_report
        results["hybrid_p_win_final"] = round(p_win_final, 4)
        results["hybrid_action"] = hybrid_action

        return results

    # ------------------------------------------------------------------
    # POMOCNICZE METODY TIER 6
    # ------------------------------------------------------------------

    def register_team_pair_similarity(self, anchor: str, positive: str,
                                       negatives: List[str]) -> Dict:
        """TIER 6.6 — krok contrastive learning dla embeddingów drużyn."""
        return self.team_embeddings.contrastive_update(anchor, positive, negatives)

    def record_temporal_match(self, team: str, feature_vector: np.ndarray):
        """TIER 6.2 — dodaje wynik meczu do historii formy drużyny."""
        self.temporal_attention.add_match(team, feature_vector)

    def evaluate_multi_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """TIER 6.9 — ocena multi-metric na żądanie (np. na zbiorze walidacyjnym)."""
        ml = self.multi_loss.evaluate(y_true, y_pred)
        self._last_multi_loss = ml
        return ml

    def hybrid_info(self) -> Dict:
        """Rozszerzona wersja ensemble_info() — TIER 1-5 + TIER 6."""
        base_info = self.ensemble_info()
        base_info["hybrid_version"] = "1.0"
        base_info["tier6_modules"] = {
            "MarketBasedCalibration": True,
            "TemporalAttention": True,
            "InjurySuspensionModel": True,
            "WeatherPitchModule": True,
            "PseudoLabeling": True,
            "ContrastiveEmbeddings": True,
            "DynamicThresholdTuning": True,
            "GPUBatchConfig": {"use_gpu": self.gpu_config.use_gpu,
                               "batch_size": self.gpu_config.batch_size},
            "MultiLossOptimization": True,
        }
        base_info["team_embeddings_known"] = len(self.team_embeddings.embeddings)
        base_info["temporal_histories_known"] = len(self.temporal_attention.team_histories)
        return base_info

    def describe(self) -> None:
        super().describe()
        print(f"{'='*65}")
        print(f"  HYBRID BETTING MODEL v1.0 — TIER 6 NADBUDOWA")
        print(f"{'='*65}")
        print(f"  6.1 Market-Based Calibration (blend={self.mbc.blend_weight}): ✓")
        print(f"  6.2 Temporal Attention (window={self.temporal_attention.window}, "
              f"dim={self.temporal_attention.feature_dim}): ✓")
        print(f"  6.3 Injury/Suspension Impact Model: ✓")
        print(f"  6.4 Weather & Pitch Impact Module: ✓")
        print(f"  6.5 Pseudo-labeling (conf>={self.pseudo_labeler.confidence_threshold}, "
              f"max_ratio={self.pseudo_labeler.max_pseudo_ratio}): ✓")
        print(f"  6.6 Contrastive Team Embeddings (dim={self.team_embeddings.embedding_dim}): ✓")
        print(f"  6.7 Dynamic Threshold Tuning: ✓")
        print(f"  6.8 GPU/Batch Config (use_gpu={self.gpu_config.use_gpu}, "
              f"batch={self.gpu_config.batch_size}): ✓")
        print(f"  6.9 Multi-Loss Optimization (monitoring): ✓")
        print(f"{'='*65}\n")


# =============================================================================
# FABRYKA HYBRID
# =============================================================================

def create_hybrid_engine(sport: str, use_optuna: bool = True,
                          optuna_trials: int = 30,
                          **hybrid_kwargs) -> HybridBettingEngineV1:
    """Tworzy silnik HYBRID v1.0 dla danego sportu."""
    return HybridBettingEngineV1(sport, use_optuna=use_optuna,
                                  optuna_trials=optuna_trials, **hybrid_kwargs)
# =============================================================================
# DEMO HYBRID v1.0
# =============================================================================

def demo_hybrid_single(sport: str = "football", use_optuna: bool = False):
    """
    Pełne demo HYBRID v1.0 dla jednego sportu — TIER 1-5 (v4.0) + TIER 6 (nowe).
    use_optuna=False → szybkie demo bez HPO.
    """
    engine = create_hybrid_engine(sport, use_optuna=use_optuna, optuna_trials=20)
    engine.describe()

    config = SPORT_CONFIGS[sport]
    n_dim = len(config.dimensions)

    # Dane syntetyczne (oznakowane)
    np.random.seed(42)
    n = 800
    X = np.random.rand(n, n_dim)
    w = np.array(config.weights[:n_dim])
    w /= w.sum()
    y = np.clip(X @ w + np.random.randn(n) * 0.05, 0, 1)

    # Dane syntetyczne (nieoznakowane) — dla TIER 6.5 pseudo-labeling
    X_unlabeled = np.random.rand(300, n_dim)

    train_report = engine.train(X, y, n_splits=3, X_unlabeled=X_unlabeled)
    engine.compile(n_points=10, eps=1e-3)

    # Przykładowy mecz
    params = np.array([0.80, 0.75, 0.30, 0.70, 0.60, 0.55, 0.45, 0.55][:n_dim])

    if sport == "football":
        engine.update_elo("TeamA", "TeamB", 1.0)
        engine.update_elo("TeamA", "TeamC", 1.0)
        engine.update_elo("TeamA", "TeamD", 0.5)

        # TIER 6.6 — przykładowa aktualizacja embeddingów drużyn
        engine.register_team_pair_similarity("TeamA", "TeamSimilarStyle", ["TeamC", "TeamD"])

    # TIER 6 — kontekst hybrydowy
    context = HybridMatchContext(
        odds_1=[1.85, 1.90, 1.88, 1.83, 1.92],
        odds_x=[3.40, 3.50, 3.45, 3.38, 3.55],
        odds_2=[4.20, 4.10, 4.30, 4.15, 4.25],
        home_recent_form=np.array([1.0, 2, 0, 1.8, 0.6]),
        away_recent_form=np.array([0.5, 1, 1, 1.2, 0.4]),
        home_absences=[
            PlayerAbsence("Striker X", "FWD", importance=0.8, reason="injury"),
        ],
        away_absences=[
            PlayerAbsence("Defender Y", "DEF", importance=0.4, reason="suspension"),
        ],
        weather=WeatherConditions(condition="rain", temperature_c=12.0, wind_speed_kmh=15.0),
        league="premier_league",
        is_derby=False,
        day_of_week=7,
        high_stakes=False,
    )

    result = engine.predict(params, team_home="TeamA", team_away="TeamB", context=context)

    print("\n  === WYNIK PREDYKCJI HYBRID v1.0 (TIER 1-5) ===")
    print(f"  p_win (raw)        : {result['p_win_raw']}")
    print(f"  p_win (calibrated) : {result['p_win_calibrated']}")
    print(f"  p_draw             : {result['p_draw']}")
    print(f"  p_loss             : {result['p_loss']}")
    print(f"  Elo feature        : {result['elo_feature']}")
    print(f"  Kelly fraction     : {result['kelly_fraction']}")
    print(f"  Confidence         : {result['confidence']}")

    print("\n  === TIER 6 — HYBRID OVERLAY ===")
    hv1 = result["hybrid_v1"]
    if hv1.get("weather"):
        print(f"  [6.4 Weather] {hv1['weather']}")
    if hv1.get("temporal_attention"):
        print(f"  [6.2 Temporal Attention] home={hv1['temporal_attention'].get('home')}")
        print(f"  [6.2 Temporal Attention] away={hv1['temporal_attention'].get('away')}")
    print(f"  [6.3 Injury/Suspension] {hv1['injury_suspension']['application']}")
    if hv1.get("market_based_calibration"):
        print(f"  [6.1 Market-Based Calibration] {hv1['market_based_calibration']}")
    print(f"  [Final probabilities] {hv1['final_probabilities']}")
    print(f"  [6.7 Dynamic Threshold] {hv1['dynamic_threshold']}")
    print(f"  [Hybrid decision] {hv1['hybrid_decision']}")

    print(f"\n  hybrid_p_win_final : {result['hybrid_p_win_final']}")
    print(f"  hybrid_action      : {result['hybrid_action']}")

    print("\n  === TRAIN REPORT (TIER 6.5 / 6.8 / 6.9) ===")
    print(json.dumps(train_report, ensure_ascii=False, indent=2, default=str))

    print("\n  === HYBRID INFO ===")
    print(json.dumps(engine.hybrid_info(), ensure_ascii=False, indent=2))


# =============================================================================
# ENTRY POINT (HYBRID v1.0) — zachowuje również oryginalny demo v4.0
# =============================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("  HYBRID BETTING MODEL v1.0 — BANEBET PRO")
    print("  (zawiera w pełni nienaruszony Universal Betting Engine v4.0)")
    print("="*70)

    sport_arg = "football"
    optuna_flag = "--optuna" in sys.argv
    legacy_mode = "--legacy" in sys.argv

    for arg in sys.argv[1:]:
        if arg.lower() in SPORT_CONFIGS or arg.lower() == "all":
            sport_arg = arg.lower()

    if legacy_mode:
        # Uruchom oryginalne demo v4.0 (El Clasico + ewentualnie wszystkie sporty)
        print("\n[MODE] --legacy : uruchamiam oryginalne demo v4.0 bez zmian\n")

        def demo_el_clasico():
            print("\n" + "="*65)
            print("  PRZYKŁADOWY MECZ: Real Madryt vs FC Barcelona")
            print("  Estadio Santiago Bernabéu — LaLiga")
            print("="*65)

            engine = create_engine("football", use_optuna=False)

            np.random.seed(42)
            n = 800
            cfg = SPORT_CONFIGS["football"]
            n_dim = len(cfg.dimensions)
            w = np.array(cfg.weights[:n_dim]); w /= w.sum()
            X_hist = np.random.rand(n, n_dim)
            y_hist = np.clip(X_hist @ w + np.random.randn(n) * 0.05, 0, 1)
            engine.train(X_hist, y_hist, n_splits=3)
            engine.compile(n_points=10, eps=1e-3)

            engine.update_elo("Real Madryt", "Atletico",  1.0)
            engine.update_elo("Real Madryt", "Getafe",    1.0)
            engine.update_elo("Real Madryt", "Sevilla",   0.5)
            engine.update_elo("Real Madryt", "Villarreal",1.0)
            engine.update_elo("Real Madryt", "Osasuna",   0.0)
            engine.update_elo("FC Barcelona","Girona",    1.0)
            engine.update_elo("FC Barcelona","Celta Vigo",0.5)
            engine.update_elo("FC Barcelona","Betis",     1.0)
            engine.update_elo("FC Barcelona","Valencia",  0.5)
            engine.update_elo("FC Barcelona","Espanyol",  0.0)

            params = np.array([0.82, 0.78, 0.25, 0.75, 0.52, 0.80, 0.72, 0.70])

            result = engine.predict(params, team_home="Real Madryt", team_away="FC Barcelona")

            print(f"""
  WYNIKI 1X2
  ----------
  P(1) Real wygrywa : {result['p_win']:.1%}   kurs fair: {result['fair_1']:.2f}
  P(X) Remis        : {result['p_draw']:.1%}   kurs fair: {result['fair_x']:.2f}
  P(2) Barca wygrywa: {result['p_loss']:.1%}   kurs fair: {result['fair_2']:.2f}

  BANEBET v5.4 DECYZJA
  --------------------
  {result['banebet_v54'].get('action')}
""")

        demo_el_clasico()

        if sport_arg == "all":
            demo_all_sports(use_optuna=False)
        elif sport_arg in SPORT_CONFIGS:
            demo_single(sport_arg, use_optuna=optuna_flag)
    else:
        # Domyślnie: pełne demo HYBRID v1.0
        if sport_arg == "all":
            for s in SPORT_CONFIGS:
                print(f"\n\n########## SPORT: {s} ##########")
                demo_hybrid_single(s, use_optuna=False)
        else:
            demo_hybrid_single(sport_arg, use_optuna=optuna_flag)
