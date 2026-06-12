"""
=============================================================================
UNIVERSAL BETTING ENGINE v2.0 — XFAC + MEBN + EXTREME ENSEMBLE
=============================================================================
Silnik predykcyjny dla WSZYSTKICH sportów bukmacherskich.

Architektura ensemble (7 modeli):
  1. XGBoost Extreme     — gradient boosting (hist/GPU)
  2. KTBoost            — kernel + tree boosting (KernelRidgeRegressor)
  3. NGBoost            — probabilistyczny boosting (Normal dist.)
  4. GPBoost            — Gaussian Process + gradient boosting
  5. RandomForest       — residual corrector
  6. MEBN               — Bayesian Network (Bayesowska sieć zależności)
  7. XFAC Tensor Train  — ultra-szybka aproksymacja przestrzeni parametrów

SPORTY:
  Piłka nożna, Koszykówka, Hokej, Tenis, Siatkówka, Baseball, Futbol Amer.,
  Rugby, Kolarstwo, Boks/MMA, Snooker, Darts, E-sport, Wyścigi, Krykiet

INSTALACJA:
  pip install numpy scipy scikit-learn xgboost
  pip install ktboost          # KTBoost
  pip install ngboost          # NGBoost
  pip install gpboost          # GPBoost
  pip install xfacpy           # opcjonalnie - jeśli niedostępne, tryb fallback
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from scipy.special import expit  # stabilny sigmoid
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
import warnings
from collections import deque
import json
import os

# ---------------------------------------------------------------------------
# XGBoost Extreme
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost niedostępny — używam sklearn GradientBoosting jako fallback.")

# ---------------------------------------------------------------------------
# KTBoost — Kernel + Tree Boosting
# ---------------------------------------------------------------------------
try:
    import KTBoost.KTBoost as KTBoost
    KTB_AVAILABLE = True
except ImportError:
    KTB_AVAILABLE = False
    warnings.warn("KTBoost niedostępny — model pomijany w ensemble.")

# ---------------------------------------------------------------------------
# NGBoost — Natural Gradient Boosting (probabilistyczny)
# ---------------------------------------------------------------------------
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    NGB_AVAILABLE = True
except ImportError:
    NGB_AVAILABLE = False
    warnings.warn("NGBoost niedostępny — model pomijany w ensemble.")

# ---------------------------------------------------------------------------
# GPBoost — Gaussian Process + Gradient Boosting
# ---------------------------------------------------------------------------
try:
    import gpboost as gpb
    GPB_AVAILABLE = True
except ImportError:
    GPB_AVAILABLE = False
    warnings.warn("GPBoost niedostępny — model pomijany w ensemble.")

# ---------------------------------------------------------------------------
# XFAC — Tensor Train
# ---------------------------------------------------------------------------
try:
    import xfacpy
    XFAC_AVAILABLE = True
except ImportError:
    XFAC_AVAILABLE = False
    warnings.warn("xfacpy niedostępny — predykcja działa w trybie bezpośrednim (wolniej).")


# =============================================================================
# KONFIGURACJA SPORTÓW — parametry MEBN dla każdego sportu
# =============================================================================

@dataclass
class SportConfig:
    """
    Konfiguracja parametrów MEBN dla danego sportu.
    Każdy parametr to wymiar tensora TT.
    """
    name: str
    # Nazwy wymiarów (cechy kontekstu meczu)
    dimensions: List[str]
    # Wagi bayesowskie: jak bardzo każdy wymiar wpływa na wynik
    weights: List[float]
    # Typowe rynki bukmacherskie dla tego sportu
    markets: List[str]
    # Czy sport ma przerwę/sety (wpływa na zmęczenie)
    has_sets: bool = False
    # Czy pogoda istotna
    weather_relevant: bool = False


SPORT_CONFIGS: Dict[str, SportConfig] = {

    "football": SportConfig(
        name="Piłka Nożna",
        dimensions=["team_power", "home_advantage", "fatigue", "form",
                    "head2head", "attack", "defense", "pressure"],
        weights=[0.30, 0.15, 0.10, 0.20, 0.10, 0.05, 0.05, 0.05],
        markets=["1X2", "DC", "OU_2.5", "BTTS", "AH", "HT_FT",
                 "Correct_Score", "First_Goal", "Cards_OU"],
        weather_relevant=False,
    ),

    "basketball": SportConfig(
        name="Koszykówka",
        dimensions=["team_power", "home_advantage", "fatigue", "shooting_pct",
                    "pace", "injury_impact", "form", "back2back"],
        weights=[0.28, 0.12, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05],
        markets=["1X2", "Handicap", "Total_OU", "1H_OU",
                 "Race_to_X", "Player_Props", "Q1_OU"],
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
# KLASA MEBN — Bayesian Entity Fragment dla dowolnego sportu
# =============================================================================

class SportMEBN:
    """
    Multi-Entity Bayesian Network — generyczny szablon dla dowolnego sportu.
    Funkcja probability_function() jest 'czarną skrzynką' wywoływaną przez XFAC.
    """

    def __init__(self, config: SportConfig, tree_weights: Optional[Tuple[float, float]] = None):
        self.config = config
        n = len(config.dimensions)
        self.w = np.array(config.weights[:n])
        self.w /= self.w.sum()  # normalizacja
        # Wagi z drzew decyzyjnych (jeśli dostępne po treningu)
        # Uwaga: NIE są używane do skalowania linear — tylko jako informacja
        self.xgb_w = tree_weights[0] if tree_weights else 0.5
        self.rf_w  = tree_weights[1] if tree_weights else 0.5
        # Kalibracja: dla neutralnego wektora [0.5]*n zwracamy p=0.5
        # synergy i pressure_term zawsze dodają wartość dodatnią → bias w górę bez korekty
        _v0 = np.array([0.5] * n)
        _lin0  = float(np.dot(self.w, _v0))
        _syn0  = sum(_v0[i]*_v0[i+1]*self.w[i]*self.w[i+1] for i in range(n-1))
        _pt0   = float(np.sin(_v0[-1] * np.pi) * 0.05)
        self._neutral_offset = _lin0 + _syn0 * 0.5 + _pt0  # total dla neutralnego wektora

    def probability_function(self, coords: np.ndarray) -> float:
        """
        Funkcja wywoływana przez xfacpy.cross().
        coords: wektor o długości len(dimensions), wartości w [0, 1].
        Zwraca: prawdopodobieństwo wygranej fav. strony [0, 1].
        """
        n = len(self.config.dimensions)
        coords = np.clip(coords[:n], 0.0, 1.0)

        # Liniowa kombinacja Bayesowska z wagami
        linear = float(np.dot(self.w, coords))

        # Korekcja nieline arowa — dynamiczna interakcja parametrów
        # (symuluje efekt synergii/antagonizmu cech)
        synergy = 0.0
        for i in range(n - 1):
            synergy += coords[i] * coords[i + 1] * self.w[i] * self.w[i + 1]

        # Składnik presji (ostatnia cecha jako "presja/zmienność")
        volatility = coords[-1]
        pressure_term = np.sin(volatility * np.pi) * 0.05

        # base = linear (bez skalowania przez wagi drzew — to powodowało away bias)
        # wagi drzew są używane tylko do inicjalizacji wag MEBN przez TreeLayer
        total = linear + synergy * 0.5 + pressure_term

        # Odejmij neutral_offset: neutral [0.5]*n → p=0.5 (brak home/away bias)
        return float(np.clip(expit((total - self._neutral_offset) * 5.0), 0.0, 1.0))


# =============================================================================
# WARSTWA DRZEW DECYZYJNYCH — Extreme Ensemble (5 modeli)
# =============================================================================

class TreeLayer:
    """
    Extreme Ensemble 7-modelowy:
      1. XGBoost Extreme   — główny booster (hist/GPU)
      2. KTBoost           — kernel + tree, łapie nieliniowości
      3. NGBoost           — probabilistyczny, zwraca rozkład Normal
      4. GPBoost           — Gaussian Process + boosting, niepewność bayesowska
      5. RandomForest      — korektor residuów XGB
    
    Predykcja finalna: ważona średnia dostępnych modeli.
    Wagi ensemble obliczane na podstawie OOF MAE (odwrotność błędu).
    """

    # Wagi domyślne (gdy nie ma walidacji OOF)
    _DEFAULT_WEIGHTS = {
        "xgb": 0.30,
        "ktb": 0.20,
        "ngb": 0.20,
        "gpb": 0.15,
        "rf":  0.15,
    }

    def __init__(self):
        # --- XGBoost Extreme ---
        if XGB_AVAILABLE:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:absoluteerror",
                tree_method="hist",
                verbosity=0,
            )
        else:
            self.xgb_model = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
            )

        # --- KTBoost ---
        if KTB_AVAILABLE:
            self.ktb_model = KTBoost.BoostingRegressor(
                loss="mse",
                n_estimators=200,
                learning_rate=0.05,
                base_learner="kernel",      # kernel + tree hybrid
                kernel="rbf",
                gamma=None,                 # auto
            )
        else:
            self.ktb_model = None

        # --- NGBoost ---
        if NGB_AVAILABLE:
            self.ngb_model = NGBRegressor(
                Dist=Normal,
                n_estimators=200,
                learning_rate=0.05,
                verbose=False,
                random_state=42,
            )
        else:
            self.ngb_model = None

        # --- GPBoost ---
        if GPB_AVAILABLE:
            self.gpb_model = None          # inicjowany w train() (wymaga danych)
            self._gpb_params = {
                "num_iterations": 200,
                "learning_rate":  0.05,
                "max_depth":      5,
                "min_data_in_leaf": 5,
                "objective":      "regression_l1",
                "verbose":        -1,
            }
        else:
            self.gpb_model = None

        # --- RandomForest (residual corrector) ---
        self.rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=8, bootstrap=True,
            n_jobs=-1, random_state=42,
        )

        self.trained = False
        self._ensemble_weights: Dict[str, float] = dict(self._DEFAULT_WEIGHTS)
        # feature importances (śr. z dostępnych modeli)
        self._xgb_importance: float = 0.5
        self._rf_importance:  float = 0.5

    # ------------------------------------------------------------------
    # TRENING
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
        """
        K-fold OOF dla wszystkich modeli → obliczenie wag ensemble.
        RF trenowany na residuach XGB (główna korekcja).
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        n = len(y)

        oof: Dict[str, np.ndarray] = {
            "xgb": np.zeros(n),
            "ktb": np.zeros(n),
            "ngb": np.zeros(n),
            "gpb": np.zeros(n),
        }
        active = {
            "xgb": True,
            "ktb": KTB_AVAILABLE and self.ktb_model is not None,
            "ngb": NGB_AVAILABLE and self.ngb_model is not None,
            "gpb": GPB_AVAILABLE,
        }

        print(f"[ENSEMBLE] Aktywne modele: "
              f"XGB={active['xgb']} | KTB={active['ktb']} | "
              f"NGB={active['ngb']} | GPB={active['gpb']} | RF=True")

        # ---- OOF loop ----
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            X_t, X_v = X[tr_idx], X[val_idx]
            y_t = y[tr_idx]

            # XGBoost
            self.xgb_model.fit(X_t, y_t)
            oof["xgb"][val_idx] = np.clip(self.xgb_model.predict(X_v), 0, 1)

            # KTBoost
            if active["ktb"]:
                try:
                    self.ktb_model.fit(X_t, y_t)
                    oof["ktb"][val_idx] = np.clip(self.ktb_model.predict(X_v), 0, 1)
                except Exception as e:
                    warnings.warn(f"[KTBoost fold {fold}] {e}")
                    active["ktb"] = False

            # NGBoost
            if active["ngb"]:
                try:
                    self.ngb_model.fit(X_t, y_t)
                    # predict zwraca rozkład — bierzemy loc (mean)
                    oof["ngb"][val_idx] = np.clip(
                        self.ngb_model.predict(X_v), 0, 1
                    )
                except Exception as e:
                    warnings.warn(f"[NGBoost fold {fold}] {e}")
                    active["ngb"] = False

            # GPBoost
            if active["gpb"]:
                try:
                    gpb_data = gpb.Dataset(X_t, label=y_t)
                    _gp_model = gpb.GPModel(num_data=len(y_t), likelihood="gaussian")
                    _gpb_booster = gpb.train(
                        params=self._gpb_params,
                        train_set=gpb_data,
                        gp_model=_gp_model,
                        num_boost_round=self._gpb_params["num_iterations"],
                        valid_sets=None,
                    )
                    gpb_pred = _gpb_booster.predict(
                        data=X_v, gp_coords_pred=X_v,
                        predict_var=False,
                    )
                    # predict może zwrócić dict lub array
                    if isinstance(gpb_pred, dict):
                        gpb_pred = gpb_pred.get("response_mean", list(gpb_pred.values())[0])
                    oof["gpb"][val_idx] = np.clip(gpb_pred, 0, 1)
                except Exception as e:
                    warnings.warn(f"[GPBoost fold {fold}] {e}")
                    active["gpb"] = False

        # ---- Obliczenie wag OOF (odwrotność MAE) ----
        weights: Dict[str, float] = {}
        for key, pred in oof.items():
            if not active[key]:
                weights[key] = 0.0
                continue
            mae = float(np.mean(np.abs(pred - y))) + 1e-6
            weights[key] = 1.0 / mae

        # RF trenowany na residuach XGB — jego wagę liczymy z MAE residuów OOF
        oof_residuals = y - oof["xgb"]  # residua XGB na OOF
        mae_rf_residual = float(np.mean(np.abs(oof_residuals))) + 1e-6
        weights["rf"] = 1.0 / mae_rf_residual

        # Normalizacja
        total_w = sum(weights.values()) + 1e-9
        self._ensemble_weights = {k: v / total_w for k, v in weights.items()}
        print(f"[ENSEMBLE] Wagi OOF: " +
              " | ".join(f"{k}={v:.3f}" for k, v in self._ensemble_weights.items()))

        # ---- Trening finalny na całym zbiorze ----
        self.xgb_model.fit(X, y)

        if active["ktb"]:
            try:
                self.ktb_model.fit(X, y)
            except Exception as e:
                warnings.warn(f"[KTBoost final] {e}")
                self._ensemble_weights["ktb"] = 0.0

        if active["ngb"]:
            try:
                self.ngb_model.fit(X, y)
            except Exception as e:
                warnings.warn(f"[NGBoost final] {e}")
                self._ensemble_weights["ngb"] = 0.0

        if active["gpb"]:
            try:
                gpb_data_full = gpb.Dataset(X, label=y)
                _gp_full = gpb.GPModel(num_data=len(y), likelihood="gaussian")
                self.gpb_model = gpb.train(
                    params=self._gpb_params,
                    train_set=gpb_data_full,
                    gp_model=_gp_full,
                    num_boost_round=self._gpb_params["num_iterations"],
                    valid_sets=None,
                )
            except Exception as e:
                warnings.warn(f"[GPBoost final] {e}")
                self._ensemble_weights["gpb"] = 0.0

        # RF na residuach XGB
        xgb_preds_train = np.clip(self.xgb_model.predict(X), 0, 1)
        residuals = y - xgb_preds_train
        self.rf_model.fit(X, residuals)

        # Feature importances → do MEBN
        self._xgb_importance = float(self.xgb_model.feature_importances_.mean())
        self._rf_importance   = float(self.rf_model.feature_importances_.mean())

        self.trained = True
        print("[ENSEMBLE] Trening zakończony.")

    # ------------------------------------------------------------------
    # PREDYKCJA
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ważona predykcja ensemble wszystkich modeli."""
        assert self.trained, "Wywołaj train() przed predict()."

        # Baza: XGBoost + korekcja RF residuów (jak w karasu2 — RF liczony raz)
        xgb_pred = np.clip(self.xgb_model.predict(X), 0, 1)
        res_pred = self.rf_model.predict(X)
        base_pred = np.clip(xgb_pred + res_pred * self._ensemble_weights["rf"], 0, 1)

        # Modele dodatkowe (KTB/NGB/GPB) — ważona suma z base_pred
        extra_preds: Dict[str, np.ndarray] = {}

        if KTB_AVAILABLE and self.ktb_model is not None and self._ensemble_weights.get("ktb", 0) > 0:
            try:
                extra_preds["ktb"] = np.clip(self.ktb_model.predict(X), 0, 1)
            except Exception:
                pass

        if NGB_AVAILABLE and self.ngb_model is not None and self._ensemble_weights.get("ngb", 0) > 0:
            try:
                extra_preds["ngb"] = np.clip(self.ngb_model.predict(X), 0, 1)
            except Exception:
                pass

        if GPB_AVAILABLE and self.gpb_model is not None and self._ensemble_weights.get("gpb", 0) > 0:
            try:
                gpb_pred = self.gpb_model.predict(data=X, gp_coords_pred=X, predict_var=False)
                if isinstance(gpb_pred, dict):
                    gpb_pred = gpb_pred.get("response_mean", list(gpb_pred.values())[0])
                extra_preds["gpb"] = np.clip(gpb_pred, 0, 1)
            except Exception:
                pass

        if not extra_preds:
            # Brak modeli dodatkowych — identycznie jak karasu2
            return base_pred

        # Ważona kombinacja base (xgb+rf) z modelami dodatkowymi
        w_base = self._ensemble_weights.get("xgb", 0.5)
        result = base_pred * w_base
        total_w = w_base
        for key, pred in extra_preds.items():
            w = self._ensemble_weights.get(key, 0.0)
            if w > 0:
                result += pred * w
                total_w += w

        if total_w > 0:
            result /= total_w

        return np.clip(result, 0.0, 1.0)

    # ------------------------------------------------------------------
    # NGBoost — dodatkowe API: rozkład niepewności
    # ------------------------------------------------------------------

    def predict_uncertainty(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Zwraca odchylenie standardowe predykcji (z NGBoost).
        None jeśli NGBoost niedostępny.
        Użyj do oceny pewności zakładu.
        """
        if not (NGB_AVAILABLE and self.ngb_model is not None and self.trained):
            return None
        try:
            dist = self.ngb_model.pred_dist(X)
            return np.array(dist.scale)  # std z rozkładu Normal
        except Exception:
            return None

    # ------------------------------------------------------------------
    # WŁAŚCIWOŚCI
    # ------------------------------------------------------------------

    @property
    def tree_weights(self) -> Tuple[float, float]:
        """Zwraca (xgb_importance, rf_importance) dla MEBN."""
        if not self.trained:
            return (0.5, 0.5)
        total = self._xgb_importance + self._rf_importance + 1e-9
        return (self._xgb_importance / total, self._rf_importance / total)

    @property
    def ensemble_weights(self) -> Dict[str, float]:
        """Aktualne wagi ensemble po treningu."""
        return dict(self._ensemble_weights)

    @property
    def available_models(self) -> List[str]:
        """Lista aktywnych modeli w ensemble."""
        models = ["XGBoost Extreme", "RandomForest"]
        if KTB_AVAILABLE: models.append("KTBoost")
        if NGB_AVAILABLE: models.append("NGBoost")
        if GPB_AVAILABLE: models.append("GPBoost")
        return models


# =============================================================================
# ADAPTACYJNY UCZEŃ — Bayesian Online Learning dla MAŁEJ liczby meczów
# =============================================================================

class AdaptiveLearner:
    """
    Mechanizm adaptacyjnego uczenia się na MAŁEJ liczbie meczów (10-30).
    NIE ZMIENIA architektury 8 wymiarów – tylko DOSTRAJA wagi MEBN.
    
    Zasada działania:
    1. Zaczyna od wag eksperckich (z SportConfig)
    2. Każdy nowy mecz to jeden epizod uczenia
    3. Bayesian update: nowa_waga = (stara_waga * alpha + error * beta) / (alpha + beta)
    4. Im więcej meczów, tym większe zaufanie do korekty
    """
    
    def __init__(self, engine: 'UniversalBettingEngine', learning_rate: float = 0.05):
        self.engine = engine
        self.learning_rate = learning_rate
        
        # Pamięć meczów (parametry + rzeczywisty wynik)
        self.match_history: List[Tuple[np.ndarray, float]] = []
        
        # Adaptacyjne korekty wag dla każdego wymiaru (start = brak korekty)
        self.weight_corrections: Dict[str, float] = {
            dim: 0.0 for dim in engine.config.dimensions
        }
        
        # Bayesian prior – im większy, tym wolniej model się uczy (bardziej ufa ekspertowi)
        self.bayesian_alpha: float = 20.0  # zaufanie do eksperta
        self.bayesian_beta: float = 1.0    # zaufanie do nowych danych (rośnie z każdym meczem)
        
        # Śledzenie błędów
        self.prediction_errors: List[float] = []
        
        print(f"[ADAPTIVE] Inicjalizacja adaptacyjnego uczenia dla {engine.config.name}")
        print(f"[ADAPTIVE] learning_rate={learning_rate}, bayesian_alpha={self.bayesian_alpha}")
    
    def record_match(self, params: np.ndarray, actual_result: float) -> Dict:
        """
        Zapisuje wynik meczu i aktualizuje wagi.
        
        Args:
            params: parametry meczu (8 wymiarów, wartości 0-1)
            actual_result: rzeczywisty wynik (0 = przegrana, 1 = wygrana)
        
        Returns:
            słownik z informacją o korekcie
        """
        # Predykcja przed korektą
        predicted = self.engine.predict(params)["p_win"]
        error = actual_result - predicted
        
        self.match_history.append((params.copy(), actual_result))
        self.prediction_errors.append(abs(error))
        
        # Bayesian update – aktualizacja zaufania do nowych danych
        self.bayesian_beta += 1.0
        
        # Obliczenie korekty dla każdego wymiaru
        corrections = {}
        for i, dim in enumerate(self.engine.config.dimensions):
            # Wpływ parametru na błąd (im wyższy parametr, tym większa korekta jeśli błąd duży)
            param_value = params[i]
            
            # Bayesian weight – im więcej danych, tym większe zaufanie do korekty
            bayesian_weight = self.bayesian_beta / (self.bayesian_alpha + self.bayesian_beta)
            
            # Korekta = error * param_value * learning_rate * bayesian_weight
            correction = error * param_value * self.learning_rate * bayesian_weight
            
            # Ograniczenie korekty do rozsądnych wartości
            correction = np.clip(correction, -0.05, 0.05)
            
            self.weight_corrections[dim] += correction
            corrections[dim] = correction
        
        # Aktualizacja wag w MEBN
        self._apply_corrections()
        
        # Obliczenie nowej predykcji po korekcie
        new_predicted = self.engine.predict(params)["p_win"]
        
        return {
            "dimension_corrections": corrections,
            "old_prediction": round(predicted, 4),
            "actual_result": actual_result,
            "error": round(error, 4),
            "new_prediction": round(new_predicted, 4),
            "matches_learned": len(self.match_history),
            "avg_error": round(np.mean(self.prediction_errors[-10:]), 4) if self.prediction_errors else 0,
        }
    
    def _apply_corrections(self):
        """Aplikuje skorygowane wagi do MEBN."""
        n = len(self.engine.config.dimensions)
        original_weights = np.array(self.engine.config.weights[:n])
        
        # Zastosowanie korekt
        corrected_weights = original_weights.copy()
        for i, dim in enumerate(self.engine.config.dimensions):
            correction = self.weight_corrections[dim]
            corrected_weights[i] = original_weights[i] * (1.0 + correction)
        
        # Normalizacja (suma = 1)
        corrected_weights = np.maximum(corrected_weights, 0.01)  # żadna waga nie może być ujemna
        corrected_weights /= corrected_weights.sum()
        
        # Aktualizacja wag w MEBN
        if self.engine.mebn is not None:
            self.engine.mebn.w = corrected_weights
            
        # Zapamiętanie skorygowanych wag
        self.engine._corrected_weights = corrected_weights.tolist()
    
    def get_current_weights(self) -> Dict[str, float]:
        """Zwraca aktualne wagi (eksperckie + korekty)."""
        if self.engine.mebn is None:
            return {}
        
        result = {}
        for i, dim in enumerate(self.engine.config.dimensions):
            original = self.engine.config.weights[i]
            correction = self.weight_corrections[dim]
            result[dim] = {
                "original": original,
                "correction": round(correction, 4),
                "current": round(self.engine.mebn.w[i], 4),
            }
        return result
    
    def save_memory(self, filepath: str):
        """Zapisuje historię meczów do pliku JSON."""
        data = {
            "sport": self.engine.sport,
            "match_history": [
                {
                    "params": params.tolist(),
                    "actual_result": result,
                }
                for params, result in self.match_history
            ],
            "weight_corrections": self.weight_corrections,
            "bayesian_alpha": self.bayesian_alpha,
            "bayesian_beta": self.bayesian_beta,
            "prediction_errors": self.prediction_errors,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[ADAPTIVE] Zapisano historię do {filepath}")
    
    def load_memory(self, filepath: str):
        """Wczytuje historię meczów z pliku JSON."""
        if not os.path.exists(filepath):
            print(f"[ADAPTIVE] Brak pliku {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.match_history = [
            (np.array(m["params"]), m["actual_result"])
            for m in data["match_history"]
        ]
        self.weight_corrections = data["weight_corrections"]
        self.bayesian_alpha = data["bayesian_alpha"]
        self.bayesian_beta = data["bayesian_beta"]
        self.prediction_errors = data["prediction_errors"]
        
        self._apply_corrections()
        print(f"[ADAPTIVE] Wczytano {len(self.match_history)} meczów z {filepath}")


# =============================================================================
# GŁÓWNY SILNIK — Universal Betting Engine
# =============================================================================

class UniversalBettingEngine:
    """
    Główny silnik łączący:
    1. Warstwę drzew (XGBoost + RF) — uczy się z danych historycznych
    2. MEBN — bayesowski model zależności cech
    3. XFAC (Tensor Train) — ultra-szybka aproksymacja przestrzeni parametrów
    4. AdaptiveLearner — adaptacyjne uczenie na MAŁEJ liczbie meczów (NOWOŚĆ)
    """

    def __init__(self, sport: str):
        if sport not in SPORT_CONFIGS:
            available = ", ".join(SPORT_CONFIGS.keys())
            raise ValueError(f"Nieznany sport '{sport}'. Dostępne: {available}")

        self.sport = sport
        self.config = SPORT_CONFIGS[sport]
        self.tree_layer = TreeLayer()
        self.mebn: Optional[SportMEBN] = None
        self.tt_model = None  # Tensor Train (xfac)
        self._is_compiled = False
        self._corrected_weights: Optional[List[float]] = None  # skorygowane wagi przez AdaptiveLearner
        
        # NOWOŚĆ: adaptacyjny uczeń
        self.adaptive = None  # inicjowany po train()

        n_dim = len(self.config.dimensions)
        print(f"[ENGINE] Inicjalizacja: {self.config.name} ({n_dim} wymiarów MEBN)")
        print(f"[ENGINE] Wymiary: {self.config.dimensions}")
        print(f"[ENGINE] Rynki: {self.config.markets}")

    # ------------------------------------------------------------------
    # 1. TRENING WARSTWY DRZEW
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
        """
        X: macierz cech historycznych [n_samples, n_features]
           Cechy muszą odpowiadać dimensions z SportConfig.
        y: wektor wyników (np. power_rating zwycięzcy) [n_samples]
        """
        print(f"[TRAIN] Trenowanie warstwy drzew (K={n_splits})...")
        self.tree_layer.train(X, y, n_splits)
        xw, rw = self.tree_layer.tree_weights
        print(f"[TRAIN] Wagi: XGBoost={xw:.3f}, RF={rw:.3f}")

        # Inicjalizacja MEBN z wagami z drzew
        self.mebn = SportMEBN(self.config, tree_weights=(xw, rw))
        print("[TRAIN] MEBN zainicjalizowany.")
        
        # NOWOŚĆ: inicjalizacja adaptacyjnego ucznia
        self.adaptive = AdaptiveLearner(self)

    # ------------------------------------------------------------------
    # 2. KOMPILACJA DO TENSOR TRAIN (XFAC)
    # ------------------------------------------------------------------

    def compile(self, n_points: int = 15, eps: float = 1e-4) -> None:
        """
        Buduje aproksymację Tensor Train za pomocą xfacpy.
        Po compile() predykcja jest ~1000x szybsza niż bezpośrednie MEBN.
        """
        if self.mebn is None:
            raise RuntimeError("Wywołaj train() przed compile().")

        n = len(self.config.dimensions)
        low  = np.zeros(n)
        high = np.ones(n)

        if XFAC_AVAILABLE:
            print(f"[COMPILE] Budowanie Tensor Train (n_points={n_points}, eps={eps})...")
            self.tt_model = xfacpy.cross(
                self.mebn.probability_function,
                low,
                high,
                n_points=n_points,
                eps=eps,
            )
            self._is_compiled = True
            print("[COMPILE] Tensor Train gotowy. Predykcja live: O(r²d) zamiast O(n^d).")
        else:
            print("[COMPILE] xfacpy niedostępne — predykcja będzie używać MEBN bezpośrednio.")
            self._is_compiled = False

    # ------------------------------------------------------------------
    # 3. PREDYKCJA
    # ------------------------------------------------------------------

    def predict(self, match_params: np.ndarray) -> Dict:
        """
        Predykcja dla jednego meczu/zdarzenia.

        match_params: wektor wartości [0,1] dla każdego wymiaru w kolejności
                      jak w SportConfig.dimensions.

        Zwraca słownik z wynikami dla wszystkich rynków.
        """
        if self.mebn is None:
            raise RuntimeError("Wywołaj train() przed predict().")

        n = len(self.config.dimensions)
        if len(match_params) != n:
            raise ValueError(
                f"Oczekiwano {n} parametrów ({self.config.dimensions}), "
                f"otrzymano {len(match_params)}."
            )

        # Prawdopodobieństwo bazowe
        if self._is_compiled and self.tt_model is not None:
            p_win = float(self.tt_model.eval(match_params))
        else:
            p_win = self.mebn.probability_function(match_params)

        results = self._calculate_markets(p_win, match_params)

        # Niepewność NGBoost (jeśli dostępna)
        uncertainty = self.predict_uncertainty(match_params)
        if uncertainty is not None:
            results["ngb_uncertainty_std"] = round(uncertainty, 4)
        
        # NOWOŚĆ: dodaj informację o adaptacyjnych wagach
        if self.adaptive and self._corrected_weights:
            results["adaptive_weights_active"] = True
            results["correction_factor"] = round(
                np.mean([abs(v) for v in self.adaptive.weight_corrections.values()]), 4
            )

        return results

    def predict_batch(self, batch: np.ndarray) -> List[Dict]:
        """Predykcja dla wielu zdarzeń naraz."""
        return [self.predict(row) for row in batch]

    def predict_uncertainty(self, match_params: np.ndarray) -> Optional[float]:
        """
        Zwraca odchylenie standardowe predykcji z NGBoost.
        Im wyższe std → mniej pewna predykcja → ostrożniej z zakładem.
        None jeśli NGBoost niedostępny.
        """
        if self.mebn is None:
            return None
        X = match_params.reshape(1, -1)
        std_arr = self.tree_layer.predict_uncertainty(X)
        if std_arr is None:
            return None
        return float(std_arr[0])

    def ensemble_info(self) -> Dict:
        """Zwraca info o stanie ensemble — wagi, dostępne modele."""
        info = {
            "available_models": self.tree_layer.available_models,
            "ensemble_weights": self.tree_layer.ensemble_weights if self.tree_layer.trained else {},
            "xfac_compiled":   self._is_compiled,
            "xgb":  XGB_AVAILABLE,
            "ktb":  KTB_AVAILABLE,
            "ngb":  NGB_AVAILABLE,
            "gpb":  GPB_AVAILABLE,
            "xfac": XFAC_AVAILABLE,
        }
        
        # NOWOŚĆ: dodaj informacje o adaptacyjnym uczeniu
        if self.adaptive:
            info["adaptive_active"] = True
            info["matches_learned"] = len(self.adaptive.match_history)
            info["avg_prediction_error"] = round(np.mean(self.adaptive.prediction_errors[-10:]), 4) if self.adaptive.prediction_errors else None
            info["current_weights"] = self.adaptive.get_current_weights()
        
        return info

    # ------------------------------------------------------------------
    # 4. OBLICZANIE RYNKÓW BUKMACHERSKICH
    # ------------------------------------------------------------------

    def _calculate_markets(self, p_win: float, params: np.ndarray) -> Dict:
        """
        Na podstawie p_win (prawdopodobieństwo wygranej fav.) wylicza
        fair value dla wszystkich rynków dostępnych w danym sporcie.
        """
        p_win   = np.clip(p_win, 0.01, 0.99)
        p_draw  = self._estimate_draw(p_win, params)
        p_loss  = max(0.01, 1.0 - p_win - p_draw)

        # Normalizacja do sumy 1
        total = p_win + p_draw + p_loss
        p_win  /= total
        p_draw /= total
        p_loss /= total

        # Kursy fair value (bez marży bukmachera)
        fair_1  = 1.0 / p_win
        fair_x  = 1.0 / p_draw if p_draw > 0.005 else None
        fair_2  = 1.0 / p_loss

        # Parametry pomocnicze
        fatigue_idx = self._dim_index("fatigue", "back2back", "tiredness")
        form_idx    = self._dim_index("form", "form_last5", "current_form")
        fatigue     = float(params[fatigue_idx]) if fatigue_idx is not None else 0.3
        form        = float(params[form_idx])    if form_idx    is not None else 0.5

        # Over/Under bazowy
        p_over  = self._estimate_over(p_win, fatigue, form)
        p_btts  = self._estimate_btts(p_over, p_draw)

        # Kelly criterion — frakcja 1/8 Kelly przy założeniu 5% edge vs kursy fair
        # Formuła: f = (p*b - q) / b  gdzie b = bookie_odds - 1
        # Przy rzeczywistym kursie używaj: engine.kelly_with_odds(p_win, real_odds)
        _bookie_odds = (1.0 / p_win) * 1.05  # zakładamy 5% advantage vs fair
        _b = _bookie_odds - 1.0
        _q = 1.0 - p_win
        kelly_full = (p_win * _b - _q) / (_b + 1e-9)
        kelly_frac = max(0.0, kelly_full / 8.0)  # 1/8 Kelly

        out = {
            "sport":        self.config.name,
            "dimensions":   dict(zip(self.config.dimensions, params.tolist())),
            # --- Wyniki bazowe ---
            "p_win":        round(p_win,  4),
            "p_draw":       round(p_draw, 4),
            "p_loss":       round(p_loss, 4),
            # --- Fair value kursy ---
            "fair_1":       round(fair_1, 3),
            "fair_x":       round(fair_x, 3) if fair_x else None,
            "fair_2":       round(fair_2, 3),
            # --- Rynki specjalne ---
            "p_over":       round(p_over,  4),
            "p_under":      round(1 - p_over, 4),
            "fair_over":    round(1.0 / p_over, 3),
            "fair_under":   round(1.0 / (1 - p_over), 3),
            "p_btts_yes":   round(p_btts,       4),
            "p_btts_no":    round(1 - p_btts,   4),
            # --- Kelly ---
            "kelly_fraction": round(kelly_frac, 4),
            # --- Rynki dostępne ---
            "available_markets": self.config.markets,
            # --- Pewność sygnału ---
            "confidence": self._confidence_label(p_win),
        }

        # Double Chance
        out["p_dc_1x"] = round(p_win + p_draw, 4)
        out["p_dc_x2"] = round(p_draw + p_loss, 4)
        out["p_dc_12"] = round(p_win + p_loss, 4)

        return out

    def _estimate_draw(self, p_win: float, params: np.ndarray) -> float:
        """Estymacja prawdopodobieństwa remisu na podstawie sport + parametrów."""
        sport_draw_base = {
            "football":          0.26,
            "hockey":            0.00,  # OT/SO zamiast remisu w reg.
            "basketball":        0.00,
            "tennis":            0.00,
            "volleyball":        0.00,
            "baseball":          0.00,
            "american_football": 0.005,
            "rugby":             0.02,
            "cycling":           0.00,
            "boxing_mma":        0.05,
            "snooker":           0.00,
            "darts":             0.00,
            "esports":           0.00,
            "racing":            0.00,
            "cricket":           0.20,
        }
        base = sport_draw_base.get(self.sport, 0.0)
        # Wyższy p_win → mniejsza szansa remisu
        # Użyj tanh zamiast liniowej korekty — łagodniejsze i nie zeruje draw całkowicie
        imbalance = abs(p_win - 0.5) * 2.0   # 0 przy meczu wyrównanym, 1 przy dominacji
        draw_scale = float(np.clip(1.0 - 0.60 * imbalance, 0.20, 1.0))
        return float(np.clip(base * draw_scale, 0.0, base))

    def _estimate_over(self, p_win: float, fatigue: float, form: float) -> float:
        """Wyższe zmęczenie → mniej bramek; wyższa forma → więcej."""
        base = 0.52
        adj = form * 0.10 - fatigue * 0.08
        return float(np.clip(base + adj, 0.30, 0.75))

    def _estimate_btts(self, p_over: float, p_draw: float) -> float:
        """BTTS koreluje z Over i remisom."""
        return float(np.clip(p_over * 0.7 + p_draw * 0.5, 0.20, 0.80))

    def _dim_index(self, *names: str) -> Optional[int]:
        """Zwraca indeks wymiaru po nazwie (pierwsze trafienie)."""
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

    def kelly_with_odds(self, p_win: float, real_odds: float,
                        kelly_divisor: float = 8.0) -> float:
        """
        Oblicza stawkę Kelly z RZECZYWISTYMI kursami bukmachera.

        Args:
            p_win:         prawdopodobieństwo wygranej z modelu (0-1)
            real_odds:     kurs bukmachera (np. 1.85)
            kelly_divisor: dzielnik Kelly (domyślnie 8 = 1/8 Kelly)

        Returns:
            Frakcja bankrollu do postawienia (0-1)
        """
        if real_odds <= 1.0 or p_win <= 0:
            return 0.0
        b = real_odds - 1.0
        q = 1.0 - p_win
        kelly_full = (p_win * b - q) / (b + 1e-9)
        if kelly_full <= 0:
            return 0.0
        return round(float(np.clip(kelly_full / kelly_divisor, 0.0, 0.25)), 4)

    # ------------------------------------------------------------------
    # 5. NOWOŚĆ: ADAPTACYJNE UCZENIE DLA MAŁEJ LICZBY MECZÓW
    # ------------------------------------------------------------------
    
    def record_match_result(self, params: np.ndarray, actual_result: float) -> Dict:
        """
        Zapisuje wynik meczu i adaptacyjnie uczy się na nim.
        
        Args:
            params: parametry meczu (8 wymiarów)
            actual_result: 1 = wygrana faworyta, 0 = przegrana faworyta
        
        Returns:
            słownik z informacją o korekcie wag
        """
        if self.adaptive is None:
            raise RuntimeError("Wywołaj train() przed record_match_result().")
        
        return self.adaptive.record_match(params, actual_result)
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Zwraca aktualne wagi po adaptacyjnym uczeniu."""
        if self.adaptive is None:
            return {}
        return self.adaptive.get_current_weights()
    
    def save_adaptive_memory(self, filepath: str = None):
        """Zapisuje historię adaptacyjnego uczenia do pliku."""
        if self.adaptive is None:
            print("[ADAPTIVE] Brak aktywnego ucznia.")
            return
        if filepath is None:
            filepath = f"adaptive_memory_{self.sport}.json"
        self.adaptive.save_memory(filepath)
    
    def load_adaptive_memory(self, filepath: str = None):
        """Wczytuje historię adaptacyjnego uczenia z pliku."""
        if self.adaptive is None:
            print("[ADAPTIVE] Brak aktywnego ucznia.")
            return
        if filepath is None:
            filepath = f"adaptive_memory_{self.sport}.json"
        self.adaptive.load_memory(filepath)

    # ------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------

    def describe(self) -> None:
        print(f"\n{'='*60}")
        print(f"  SILNIK v2.0: {self.config.name.upper()}")
        print(f"{'='*60}")
        print(f"  Wymiary MEBN ({len(self.config.dimensions)}):")
        for i, (dim, w) in enumerate(zip(self.config.dimensions, self.config.weights)):
            print(f"    [{i}] {dim:<22} waga={w:.2f}")
        
        # NOWOŚĆ: pokaż skorygowane wagi jeśli istnieją
        if self._corrected_weights:
            print(f"\n  SKORYGOWANE WAGI (po adaptacyjnym uczeniu):")
            for i, (dim, w) in enumerate(zip(self.config.dimensions, self._corrected_weights)):
                original = self.config.weights[i]
                change = (w - original) / original * 100
                sign = "+" if change > 0 else ""
                print(f"    [{i}] {dim:<22} {w:.3f} (zmiana: {sign}{change:.1f}%)")
        
        print(f"\n  Rynki bukmacherskie ({len(self.config.markets)}):")
        for m in self.config.markets:
            print(f"    · {m}")
        print(f"\n  Modele ensemble:")
        print(f"    XGBoost Extreme: {'✓' if XGB_AVAILABLE else '✗ (fallback GBR)'}")
        print(f"    KTBoost:         {'✓' if KTB_AVAILABLE else '✗ (niedostępny)'}")
        print(f"    NGBoost:         {'✓' if NGB_AVAILABLE else '✗ (niedostępny)'}")
        print(f"    GPBoost:         {'✓' if GPB_AVAILABLE else '✗ (niedostępny)'}")
        print(f"    RandomForest:    ✓ (residual corrector)")
        print(f"\n  MEBN Bayesian:   ✓")
        print(f"  XFAC TT:         {'✓' if XFAC_AVAILABLE else '✗ (tryb bezpośredni)'}")
        print(f"  Skompilowany:    {'✓' if self._is_compiled else '✗'}")
        print(f"\n  ADAPTACYJNE UCZENIE: {'✓ AKTYWNE' if self.adaptive else '✗'}")
        if self.adaptive and len(self.adaptive.match_history) > 0:
            print(f"    Nauczono meczów: {len(self.adaptive.match_history)}")
            print(f"    Średni błąd (10 ost.): {np.mean(self.adaptive.prediction_errors[-10:]):.4f}" if self.adaptive.prediction_errors else "")
        if self.tree_layer.trained:
            print(f"\n  Wagi ensemble OOF:")
            for k, v in self.tree_layer.ensemble_weights.items():
                print(f"    {k:<6} = {v:.3f}")
        print(f"{'='*60}\n")


# =============================================================================
# FABRYKA — łatwe tworzenie silnika dla dowolnego sportu
# =============================================================================

def create_engine(sport: str) -> UniversalBettingEngine:
    """Tworzy i zwraca silnik dla danego sportu."""
    return UniversalBettingEngine(sport)


def list_sports() -> List[str]:
    """Zwraca listę wszystkich dostępnych sportów."""
    return list(SPORT_CONFIGS.keys())


# =============================================================================
# DEMO — uruchom dla każdego sportu
# =============================================================================

def demo_all_sports():
    print("\n" + "="*70)
    print("  UNIVERSAL BETTING ENGINE v2.0 — EXTREME ENSEMBLE + XFAC + MEBN — DEMO WSZYSTKICH SPORTÓW")
    print("="*70)

    sports_to_demo = list(SPORT_CONFIGS.keys())

    for sport_key in sports_to_demo:
        config = SPORT_CONFIGS[sport_key]
        n_dim = len(config.dimensions)

        print(f"\n{'─'*60}")
        print(f"  SPORT: {config.name.upper()}")
        print(f"{'─'*60}")

        # Inicjalizacja silnika
        engine = UniversalBettingEngine(sport_key)

        # Generujemy syntetyczne dane historyczne (n_dim cech)
        np.random.seed(42)
        n_samples = 500
        X_hist = np.random.rand(n_samples, n_dim)
        # y = wynik (silniejsza drużyna ma wyższy score)
        weights_arr = np.array(config.weights[:n_dim])
        weights_arr /= weights_arr.sum()
        y_hist = X_hist @ weights_arr + np.random.randn(n_samples) * 0.05
        y_hist = np.clip(y_hist, 0, 1)

        # Trening
        engine.train(X_hist, y_hist, n_splits=3)

        # Kompilacja TT (jeśli xfac dostępny)
        engine.compile(n_points=10, eps=1e-3)

        # Przykładowy mecz
        live_params = np.random.rand(n_dim)
        result = engine.predict(live_params)

        # Wyświetl wyniki
        print(f"\n  Parametry meczu:")
        for dim, val in result["dimensions"].items():
            print(f"    {dim:<25} = {val:.3f}")

        print(f"\n  Wyniki predykcji:")
        print(f"    p_win={result['p_win']:.4f}  p_draw={result['p_draw']:.4f}  p_loss={result['p_loss']:.4f}")
        print(f"    fair_1={result['fair_1']:.2f}  ", end="")
        if result["fair_x"]:
            print(f"fair_X={result['fair_x']:.2f}  ", end="")
        print(f"fair_2={result['fair_2']:.2f}")
        print(f"    p_over={result['p_over']:.4f}  p_btts_yes={result['p_btts_yes']:.4f}")
        print(f"    Kelly fraction={result['kelly_fraction']:.4f}")
        print(f"\n  >>> {result['confidence']}")


def demo_single(sport: str = "football"):
    """Pełne demo dla jednego sportu z describe()."""
    engine = create_engine(sport)
    engine.describe()

    config = SPORT_CONFIGS[sport]
    n_dim = len(config.dimensions)

    # Dane historyczne
    np.random.seed(0)
    X = np.random.rand(800, n_dim)
    w = np.array(config.weights[:n_dim])
    w /= w.sum()
    y = np.clip(X @ w + np.random.randn(800) * 0.05, 0, 1)

    engine.train(X, y)
    engine.compile(n_points=12, eps=1e-4)

    # Mecz przykładowy — silna drużyna domowa, mała forma, duże zmęczenie
    params = np.array([0.80, 0.75, 0.30, 0.70,
                       0.60, 0.50, 0.45, 0.55][:n_dim])
    result = engine.predict(params)

    print("\n  WYNIK PREDYKCJI:")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=4))

    print("\n  ENSEMBLE INFO:")
    print(json.dumps(engine.ensemble_info(), ensure_ascii=False, indent=4))
    
    # ===== DEMO ADAPTACYJNEGO UCZENIA =====
    print("\n" + "="*60)
    print("  DEMO ADAPTACYJNEGO UCZENIA (na 10 meczach)")
    print("="*60)
    
    # Symulacja 10 meczów z korektą
    for i in range(10):
        # Losowe parametry meczu
        match_params = np.random.rand(n_dim)
        # Symulacja rzeczywistego wyniku (z lekkim szumem)
        true_prob = engine.mebn.probability_function(match_params)
        actual = 1.0 if np.random.random() < true_prob else 0.0
        
        # Nagranie wyniku i adaptacja
        correction_info = engine.record_match_result(match_params, actual)
        
        print(f"\n  Mecz {i+1}:")
        print(f"    Predykcja przed: {correction_info['old_prediction']:.3f} → Rzeczywisty: {actual:.0f} → Błąd: {correction_info['error']:+.3f}")
        print(f"    Nowa predykcja: {correction_info['new_prediction']:.3f}")
        print(f"    Korekty wag: { {k: round(v,4) for k,v in list(correction_info['dimension_corrections'].items())[:3]} }...")
    
    print("\n  AKTUALNE WAGI PO ADAPTACJI:")
    for dim, data in engine.get_adaptive_weights().items():
        print(f"    {dim:<20}: oryginal={data['original']:.3f} → obecna={data['current']:.3f} (korekta={data['correction']:+.3f})")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sport_arg = sys.argv[1].lower()
        if sport_arg == "all":
            demo_all_sports()
        elif sport_arg in SPORT_CONFIGS:
            demo_single(sport_arg)
        else:
            print(f"Nieznany sport. Dostępne: {list_sports()}")
    else:
        # Domyślnie: demo dla piłki nożnej
        demo_single("football")
        print("\n\nAby uruchomić demo dla wszystkich sportów: python engine.py all")
        print(f"Dostępne sporty: {', '.join(list_sports())}")