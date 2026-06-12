# banebet.proANALYST
Proprietary sports betting prediction engine — architecture evolution from v1.0 to Hybrid v2.0. BUSL-1.1 licensed.

# BANEBET PRO — Architecture Evolution
### From a Single-Model Baseline to a Full Hybrid Intelligence Stack

> **Note on backtest context:** Results shown below were produced on a limited subset (~5,000 matches, 3 seasons, single-digit league count). The system is designed for 10+ seasons and 30+ leagues minimum. Numbers will improve substantially at full data scale and after proper calibration. They are shown here purely to illustrate directional progress across generations, not as production benchmarks.

---

## Generation 1 — `betting_model.py` (Internal: v1.x)

**Concept:** Prove that a unified multi-sport ensemble can outperform naive baselines.

The first generation established the core architectural principle of the entire project: a single engine capable of serving multiple sports and multiple betting markets simultaneously, rather than building isolated sport-specific models.

**Architecture highlights:**
- 7-model ensemble as the prediction backbone
- Separate `SportConfig` objects defining dimensions and weights per sport, covering football, basketball, hockey, tennis, volleyball, baseball, American football, rugby, cycling, boxing/MMA, snooker, darts, esports, motorsport, and cricket
- Tensor Train decomposition layer (XFAC) for high-dimensional parameter space approximation
- Bayesian Network component (MEBN) for dependency modeling between match variables
- All models wrapped with graceful import fallbacks — the engine runs even when optional libraries are absent
- Kelly criterion bankroll management built into the output layer
- Multi-market output: 1X2, Double Chance, BTTS, Over/Under, Asian Handicap, Correct Score

**What it lacked:** No calibration layer, no temporal awareness, no market feedback, no hyperparameter optimization. Raw ensemble weights were hand-tuned.

---

## Generation 2 — `betting_model_2_0.py` (Internal: v2.0)

**Concept:** Introduce a proprietary mathematical transformation layer on top of the ensemble.

The second generation kept the 7-model ensemble intact and added a dedicated mathematical preprocessing stack that became the signature of BANEBET PRO's internal signal processing. These layers operate between raw input features and the ensemble, shaping the probability space in a non-standard way.

**New in v2.0:**

- **CubicRationalLayer** — a cubic rational transform applied to input features before they enter the ensemble
- **DiscriminantSwitch** — a conditional branching mechanism: when a computed signal delta exceeds zero, the engine produces two prediction paths (solution+ and solution−); otherwise a single path. This gives the system a form of uncertainty-aware dual-hypothesis reasoning.
- **NegativeGammaTable** — a precomputed lookup of the Gamma function at negative half-integer values, used as a nonlinear correction factor in the Bayesian network component
- **ComplexGammaLayer** — computes |Γ(x + iy)| (modulus of the complex Gamma function) as a signal amplification weight
- **NegBinomGammaPMF** — a modified Negative Binomial PMF using the Gamma correction, applied to goal distribution modeling

These mathematical layers are not standard ML components. They encode a domain-specific probabilistic framework developed specifically for this system.

**Continuity:** All 7 ensemble members from Generation 1 are preserved unchanged. The new layers are additive.

---

## Generation 3 — `betting_model_3_0.py` (Internal: v3.0 / v4.0)

**Concept:** Industrial-grade ensemble with automated optimization and explainability.

This generation represents the most significant structural expansion. The ensemble grew from 7 to 9 members, a full calibration pipeline was introduced, hyperparameter optimization became automated, and the system gained the ability to explain its own decisions.

**New in v3.0:**

**TIER 1 — Expanded Ensemble + Calibration**
- LightGBM and CatBoost added to the base ensemble (7 → 9 models)
- Isotonic Regression calibration and Platt scaling applied post-ensemble
- `ProbabilityCalibrator` class wrapping both methods with automatic selection

**TIER 2 — Meta-Learner Stacking**
- Out-of-fold predictions from all 9 base models become features for a Ridge regression meta-learner
- Replaces the hand-weighted MAE averaging used in previous generations
- 5-fold cross-validation stacking protocol

**TIER 3 — Automated Hyperparameter Optimization + Explainability**
- Optuna integration for automated HPO (XGBoost and LightGBM tuning)
- SHAP integration: every prediction carries a `why_bet` field listing the top features driving the decision
- Falls back to built-in feature importances when SHAP is unavailable

**TIER 4 — Dynamic Elo + Monte Carlo Correct Score**
- Dynamic Elo ratings (K=32) replace static team power as a live feature input
- Elo updates after each recorded match result
- Monte Carlo simulation for Correct Score market probability distribution

**TIER 5 — Dixon-Coles Goal Model Fusion**
- Dixon-Coles bivariate Poisson model integrated for Over/Under and Correct Score markets
- Low-score correction (Dixon-Coles rho parameter) for 0-0 and 1-0 results
- Expected goals (μ_home, μ_away) computed separately and fused with ensemble output

**Mathematical layers from v2.0:** Fully preserved.

---

## Generation 4 — `hybrid_betting_model_v1.py` (Internal: v5.x modular layer)

**Concept:** Modular overlay architecture — 9 new intelligence modules built as a clean layer on top of Generation 3, without modifying anything below.

This generation introduced a deliberate architectural separation: the base engine (Generations 1–3) is treated as a stable foundation, and new capabilities are composed on top as independent, swappable modules. Each module can be disabled without affecting the rest of the system.

**TIER 6 — Hybrid Intelligence Modules:**

**6.1 — Market-Based Calibration (MBC)**
- Ingests raw odds from N bookmakers simultaneously
- Removes the bookmaker margin (overround) using multiplicative or additive (Shin) methods
- Computes a consensus implied probability as weighted median across bookmakers
- Blends the model's predicted probability with market consensus via a configurable λ parameter
- Detects value bets when the model's probability exceeds the market consensus by a defined edge threshold
- Output includes Kelly-adjusted stake sizing for detected value situations

**6.2 — Temporal Attention Mechanism**
- Stores a rolling match history per team (configurable window length)
- PyTorch LSTM with self-attention when available; falls back to exponentially-weighted numpy GRU
- The attention mechanism learns which past matches are most informative for current form
- Produces a scalar form embedding per team, capturing momentum and fatigue cycles
- Head-to-head momentum computed as a differential form signal

**6.3 — Injury and Suspension Impact Model**
- Bayesian update of win probability based on confirmed player absences
- Position-specific impact coefficients (forward > midfielder > defender > goalkeeper)
- Stacking decay: each additional absence contributes diminishing marginal impact
- Bayesian prior sharpens or softens based on historical prediction accuracy

**6.4 — Weather and Pitch Module**
- Rain, snow, heat, wind, and pitch type (natural / artificial / hybrid) as modifiers
- Affects goals multiplier, draw probability, BTTS probability, and home advantage delta
- Indoor flag disables weather effects (futsal / arena sports)
- Altitude adjustment (relevant for South American competitions)

**6.5 — Pseudo-Labeling Semi-Supervised Layer**
- Generates soft labels for unlabeled matches (matches without settled results)
- High-confidence predictions (configurable threshold) are added to the training set
- Enables the model to learn from in-progress seasons without waiting for full settlement

**6.6 — Contrastive Team Embeddings**
- Triplet loss (anchor / positive / negative) over team feature vectors
- Learns a style-similarity space: teams playing similar football cluster together
- Enables transfer learning between leagues with similar playing styles
- PCA projection for visualization and fast similarity lookup

**6.7 — Dynamic Threshold Tuner**
- The BET / NO_BET decision threshold is no longer fixed
- Adapts per league, match type (regular / derby / relegation / cup), and day of week
- Historical decision outcomes update thresholds via Bayesian posterior updates

**6.8 — GPU Acceleration and Batch Inference**
- Batch inference mode for processing multiple matches simultaneously
- GPU configuration layer for XGBoost and LightGBM tree construction
- Relevant for production API serving under concurrent load

**6.9 — Multi-Loss Optimization**
- Pseudo-Huber loss replaces MAE/MSE for robustness to outliers
- Ranking loss penalizes incorrect ordering of win probabilities
- Calibration loss (Expected Calibration Error) as an auxiliary training objective
- AUC surrogate loss (differentiable approximation) for direct AUC optimization

---

## Generation 5 — `hybrid_betting_model_v2.py` (Internal: v5.x unified)

**Concept:** Merge everything into a single self-contained engine. No external dependencies on prior version files.

Generation 5 is architecturally identical to Generation 4 in terms of capabilities (TIER 1–6 complete), but the implementation is unified into a single file. Every component from every prior generation is present and active.

**Key difference from Generation 4:**
- Generation 4 required `betting_model_4_0.py` as a base import; Generation 5 has no such dependency
- The full TIER 1–5 engine is embedded directly, making the file self-contained and deployable as a single artifact
- `HybridBettingEngineV1` inherits from `UniversalBettingEngineV3` within the same module
- Factory function `create_hybrid_engine(sport, ...)` provides a clean single-line instantiation API
- A `--legacy` flag at runtime allows running the original Generation 3 behavior for regression comparison

**Deployment target:** Docker-based prediction API, production inference endpoint.

---

## Backtest Signal — Early Results (Limited Data Regime)

> All numbers below come from ~5,000 matches across 3 seasons. This is approximately **one-fifth of the minimum recommended dataset** (10+ seasons, 30+ leagues). Numbers are directionally meaningful but not representative of full-scale performance.

### What the data shows even at this scale:

**OU35 market — the clearest signal:**

| Year | N bets | Accuracy | ROI |
|------|--------|----------|-----|
| 2023 | 55 | 58.2% | +13.8% |
| 2024 | 563 | 59.9% | +33.0% |
| 2025 | 714 | 65.1% | +32.4% |
| 2026 | 365 | 66.6% | +40.8% |

The OU35 ROI trend is upward across every measured year, with Sharpe ratio of **12.87** — a statistically meaningful signal even on limited data.

**Model vs. bookmaker accuracy — selective edge:**

| Market | Model | Bookmaker | Edge |
|--------|-------|-----------|------|
| OU35 | 63.5% | 36.6% | **+26.8 pp** |
| OU15 | 78.9% | 68.0% | **+10.9 pp** |
| BTTS | 50.1% | 45.0% | **+5.2 pp** |
| DC_12 | 73.4% | 73.1% | **+0.3 pp** |

The 1X2 market is weaker at this data scale — expected, as 1X2 prediction requires the most historical context to stabilize. The model does not outperform bookmakers on 1X2 at 5,000 matches; at 50,000+ matches across 30+ leagues, that gap is the primary calibration target.

**Validation:**
- Walk-forward MAE stability: **0.3568 ± 0.0030** — STABLE
- Overfitting gap (train vs. test MAE): **0.0005** — negligible
- Calibration verdict: WELL CALIBRATED (within the available data distribution)

---

## Architecture Summary

```
Generation 1  →  7-model ensemble, multi-sport, multi-market baseline
     ↓
Generation 2  →  + Proprietary mathematical transformation layers
     ↓
Generation 3  →  + 9-model ensemble, calibration, stacking meta-learner,
                   Optuna HPO, SHAP explainability, Dynamic Elo,
                   Monte Carlo CS, Dixon-Coles goal model
     ↓
Generation 4  →  + 9 modular hybrid overlays (market calibration,
                   temporal attention, injury model, weather,
                   semi-supervised learning, contrastive embeddings,
                   dynamic thresholds, GPU batch inference, multi-loss)
     ↓
Generation 5  →  Unified self-contained engine (all tiers, single file,
                   production-ready Docker deployment target)
```

**Sports coverage across all generations:** Football, basketball, hockey, tennis, volleyball, baseball, American football, rugby, boxing/MMA, snooker, darts, esports (CS2, LoL, Dota2, Valorant), motorsport, cricket, handball.

**Betting markets across all generations:** 1X2, Double Chance (1X / X2 / 12), BTTS, Over/Under (1.5 / 2.5 / 3.5 / 4.5), Asian Handicap, Correct Score, and extensions.

---

*BANEBET PRO is a proprietary system. This document describes the public architectural evolution. Internal mathematical frameworks, signal classification logic, decision constants, and calibration parameters are not disclosed.*

