2026-06-14 16:24:39,204 [INFO] ======================================================================
2026-06-14 16:24:39,204 [INFO]   ATOMIC HYBRID 7.0 (FOOTBALL, 1X2/DC/AH/BTTS/OU/CS) - BACKTEST START
2026-06-14 16:24:39,205 [INFO] ======================================================================
2026-06-14 16:24:39,206 [INFO] Laduje dane: /mnt/c/Users/rockd/Desktop/universal_model/merged_training_full.csv
2026-06-14 16:24:39,514 [WARNING] Brak kolumny 'status' - pomijam filtr, uzyje wszystkich wierszy.
2026-06-14 16:24:39,587 [INFO] Dane: 25075 zakonczonych meczow, 2010-08-14 - 2026-03-15
2026-06-14 16:24:39,594 [INFO] Buduje cechy Elo (walk-forward)...
2026-06-14 16:24:40,203 [INFO] Buduje cechy formy i goli (window=10, walk-forward)...
2026-06-14 16:24:41,719 [INFO] Buduje rozszerzone cechy  (walk-forward)...
2026-06-14 16:24:52,181 [INFO]   Rozszerzone cechy: 49 nowych kolumn 
2026-06-14 16:24:52,226 [INFO] Audyt data leakage...
2026-06-14 16:24:52,227 [INFO]   Leakage-free: True
2026-06-14 16:24:52,272 [INFO]   FIT:     22840 meczow (2010-08-14-2024-12-30)
2026-06-14 16:24:52,273 [INFO]   TUNE:    1066 meczow (2025-01-01-2025-08-31)
2026-06-14 16:24:52,273 [INFO]   HOLDOUT: 1169 meczow (2025-09-12-2026-03-15)
2026-06-14 16:24:52,274 [INFO]
TRENING MODELI (FIT) ...
2026-06-14 16:24:52,274 [INFO]   Trenuje: 1X2_H (y_1x2_h), features=60
2026-06-14 16:25:08,971 [INFO]   [1X2_H] Wytrenowany, n=22840
2026-06-14 16:25:09,007 [INFO]   Trenuje: 1X2_D (y_1x2_d), features=60
2026-06-14 16:25:20,349 [INFO]   [1X2_D] Wytrenowany, n=22840
2026-06-14 16:25:20,385 [INFO]   Trenuje: 1X2_A (y_1x2_a), features=60
2026-06-14 16:25:31,946 [INFO]   [1X2_A] Wytrenowany, n=22840
2026-06-14 16:25:31,981 [INFO]   Trenuje: DC1X (y_dc1x), features=60
2026-06-14 16:25:43,559 [INFO]   [DC1X] Wytrenowany, n=22840
2026-06-14 16:25:43,592 [INFO]   Trenuje: DC12 (y_dc12), features=60
2026-06-14 16:25:54,944 [INFO]   [DC12] Wytrenowany, n=22840
2026-06-14 16:25:54,978 [INFO]   Trenuje: DCX2 (y_dcx2), features=60
2026-06-14 16:26:06,374 [INFO]   [DCX2] Wytrenowany, n=22840
2026-06-14 16:26:06,407 [INFO]   Trenuje: AH05_H (y_ah05_h), features=60
2026-06-14 16:26:17,853 [INFO]   [AH05_H] Wytrenowany, n=22840
2026-06-14 16:26:17,886 [INFO]   Trenuje: AH05_A (y_ah05_a), features=60
2026-06-14 16:26:29,372 [INFO]   [AH05_A] Wytrenowany, n=22840
2026-06-14 16:26:29,404 [INFO]   Trenuje: BTTS_Y (y_btts_y), features=60
2026-06-14 16:26:40,956 [INFO]   [BTTS_Y] Wytrenowany, n=22840
2026-06-14 16:26:40,987 [INFO]   Trenuje: BTTS_N (y_btts_n), features=60
2026-06-14 16:26:52,556 [INFO]   [BTTS_N] Wytrenowany, n=22840
2026-06-14 16:26:52,590 [INFO]   Trenuje: OU15 (y_ou15_over), features=60
2026-06-14 16:27:04,180 [INFO]   [OU15] Wytrenowany, n=22840
2026-06-14 16:27:04,219 [INFO]   Trenuje: OU25 (y_ou25_over), features=60
2026-06-14 16:27:15,779 [INFO]   [OU25] Wytrenowany, n=22840
2026-06-14 16:27:15,814 [INFO]   Trenuje: OU35 (y_ou35_over), features=60
2026-06-14 16:27:27,302 [INFO]   [OU35] Wytrenowany, n=22840
2026-06-14 16:27:27,442 [INFO]   Modele zapisane: /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_model_state.pkl
2026-06-14 16:27:27,443 [INFO]
DIXON-COLES CS PROBS ...
2026-06-14 16:27:27,443 [INFO]   Dixon-Coles CS: obliczam p_model dla 12 wynikow...
2026-06-14 16:28:09,688 [INFO]   Dixon-Coles CS: obliczam p_model dla 12 wynikow...
2026-06-14 16:28:11,679 [INFO]   Dixon-Coles CS: obliczam p_model dla 12 wynikow...
2026-06-14 16:28:13,819 [INFO]   CS probs gotowe dla 12 wynikow
2026-06-14 16:28:13,820 [INFO]
THRESHOLD SWEEP NA TUNE ...
2026-06-14 16:28:13,821 [INFO] Threshold sweep na TUNE (grid=10 wartosci, wszystkie > 0.55)...
2026-06-14 16:28:15,735 [INFO]   Optymalny bet_threshold (wg Sharpe, na TUNE): 0.88
2026-06-14 16:28:15,736 [INFO]   Wybrany bet_threshold: 0.88
2026-06-14 16:28:15,736 [INFO]
KELLY SWEEP NA TUNE ...
2026-06-14 16:28:15,737 [INFO] Kelly sweep na zbiorze TUNE...
2026-06-14 16:28:16,860 [INFO]   Optymalny kelly_divisor (wg Sharpe): K/10
2026-06-14 16:28:16,861 [INFO]   Wybrany Kelly divisor: K/10
2026-06-14 16:28:16,862 [INFO]
IN-SAMPLE (FIT) - tylko diagnostyka ...
2026-06-14 16:28:20,171 [INFO]   FIT -> n=99309, hit=0.1100, roi=1055200925851377152.00%
2026-06-14 16:28:20,172 [INFO]
TUNE ...
2026-06-14 16:28:20,351 [INFO]   TUNE -> n=4585, hit=0.1051, roi=467.43%
2026-06-14 16:28:20,352 [INFO]
HOLDOUT (jedyne wiarygodne wyniki) ...
2026-06-14 16:28:20,550 [INFO]   HOLDOUT -> n=5008, hit=0.1092, roi=328.85%, sharpe=3.439, MaxDD=31.30%
2026-06-14 16:28:20,583 [INFO] Wykrywanie overfittingu...
2026-06-14 16:28:20,583 [INFO]   Hit gap (FIT->HOLD): 0.0008 (prog: 0.25) | ROI gap: 1055200925851376768.00% (prog: 15.0%)
2026-06-14 16:28:20,584 [INFO]   Overfitting: NISKI
2026-06-14 16:28:20,585 [INFO] Analiza stabilnosci (window=6 mies.)...
2026-06-14 16:28:20,592 [INFO] Sprawdzam czynniki ryzyka...
2026-06-14 16:28:20,592 [INFO]   Checks: 12 | Errors: 0 | Warnings: 0
2026-06-14 16:28:20,593 [INFO]
Zapisuje wyniki...
2026-06-14 16:28:20,852 [INFO]   Bets log CSV: /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_bets_log.csv (5008 zakladow)

======================================================================
ATOMIC HYBRID 42 (FOOTBALL, 1X2/DC/AH/BTTS/OU/CS) - BACKTEST SUMMARY
Wygenerowany: 2026-06-14 16:28
======================================================================

OKNA CZASOWE:
  FIT:     do 2024-12-31  (22840 meczow)
  TUNE:    2024-12-31-2025-08-31  (1066 meczow)
  HOLDOUT: 2025-08-31-koniec  (1169 meczow)

WYBRANY bet_threshold (sweep na TUNE, > 0.55): 0.88

WYNIKI HOLDOUT (wiarygodne):
  N zakladow:     5008  OK
  Hit rate:       0.1092
  ROI (equity):   328.85%
  Max Drawdown:   31.30%
  Sharpe:         3.439
  Brier score:    0.07013
  Koncowy bankroll: 4288.53 PLN

PORÓWNANIE OKIEN:
  FIT   -> n=99309 | hit=0.1100 | roi=1055200925851377152.00%
  TUNE  -> n=4585 | hit=0.1051 | roi=467.43%
  HOLD  -> n=5008 | hit=0.1092 | roi=328.85%

OVERFITTING:
  Poziom: NISKI
  Hit gap (FIT->HOLD):  0.0008  (prog: 0.25)
  ROI gap (FIT->HOLD):  1055200925851376768.00%  (prog: 15.0%)

DATA LEAKAGE:
  Status: BRAK

STABILNOSC:
  Verdict: STABILNY
  ROI std miedzy oknami: 0.0%

ROI PER RYNEK:
  CS_1_1     n=1073 | hit=0.1202 | roi=7.95% OK
  CS_2_2     n= 847 | hit=0.0496 | roi=-18.49% OK
  CS_1_2     n= 743 | hit=0.0767 | roi=13.23% OK
  CS_2_1     n= 576 | hit=0.1198 | roi=54.46% OK
  CS_3_1     n= 189 | hit=0.0370 | roi=-52.60% OK
  CS_0_0     n= 450 | hit=0.0333 | roi=-62.45% OK
  CS_0_2     n= 389 | hit=0.0514 | roi=-12.64% OK
  CS_1_0     n= 102 | hit=0.0882 | roi=6.27% OK
  CS_2_0     n= 127 | hit=0.0551 | roi=-55.52% OK
  OU35_U     n=  46 | hit=0.9130 | roi=153.23% NIEWIARYGODNE
  CS_3_0     n=  19 | hit=0.0000 | roi=-100.00% NIEWIARYGODNE
  CS_0_1     n= 194 | hit=0.0515 | roi=-38.93% OK
  DC12       n=  42 | hit=0.9286 | roi=20.68% NIEWIARYGODNE
  OU15_O     n=  51 | hit=0.9216 | roi=15.77% OK
  DC1X       n=  55 | hit=0.9273 | roi=-4.12% OK
  CS_0_3     n= 104 | hit=0.0192 | roi=-68.80% OK
  DCX2       n=   1 | hit=1.0000 | roi=14.99% NIEWIARYGODNE

ROI PER ROK:
  2025: n=2824 | hit=0.1048 | roi=5.40%
  2026: n=2184 | hit=0.1149 | roi=14.42%

KELLY SWEEP (TUNE):
  Optymalny: K/10 (K/10)
  K/2: roi=16199.88% sharpe=1.551 MaxDD=80.61% n=4585
  K/3: roi=7539.39% sharpe=2.175 MaxDD=65.08% n=4585
  K/4: roi=3654.96% sharpe=2.585 MaxDD=53.85% n=4585
  K/5: roi=2088.91% sharpe=2.893 MaxDD=45.70% n=4585
  K/6: roi=1354.05% sharpe=3.135 MaxDD=39.62% n=4585
  K/8: roi=723.51% sharpe=3.486 MaxDD=31.22% n=4585
  K/10: roi=467.43% sharpe=3.725 MaxDD=25.74% n=4585 <- WYBRANY

CZYNNIKI RYZYKA:
  OK: roi_from_equity_curve
  OK: hard_stop_bankroll
  OK: per_market_thresholds
  OK: min_bets_filter
  OK: data_leakage
  OK: walkforward_features
  OK: sharpe_annualization
  OK: three_window_split
  OK: overfitting
  OK: kelly_on_calibrated_model
  OK: temporal_stability
  OK: bet_threshold_above_055

Checks OK: 11/11
Errors:    []
Warnings:  []

PLIKI WYJSCIOWE:
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_backtest_results.json
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_backtest_summary.txt
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_validation.json
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_vs_bookmaker.json
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_backtest_report.md
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_tracker.txt
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_model_state.pkl
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_backtest.log
  /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_bets_log.csv

Czas wykonania: 221.6s
======================================================================
2026-06-14 16:28:20,894 [INFO]
Backtest zakonczony w 221.6s
2026-06-14 16:28:20,894 [INFO]   Raport MD: /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_backtest_report.md
2026-06-14 16:28:20,894 [INFO]   Summary:   /mnt/c/Users/rockd/Desktop/universal_model/atomic_hybrid_4200_backtest_summary.txt
