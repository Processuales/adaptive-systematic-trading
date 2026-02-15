# Step 2 Reference

This file documents:
- every current Python script in the repo,
- every important output under `step2_out`,
- which scripts generate which files.

---

## Python Scripts (Current)

### Core Step 2 scripts
1. `step2_run_all.py`
- Purpose: one-click orchestrator for the full Step 2 pipeline.
- Runs: dataset build, Step 2.5 analysis, candidate selection, capital visuals, dual portfolio tests.
- Typical usage: `python step2_run_all.py`

2. `step2_build_events_dataset.py`
- Purpose: build bar features + event-level dataset from cleaned bars.
- Input: `data_clean/*_1h_rth_clean.parquet`
- Output:
  - `step2_out/bar_features/*.parquet`
  - `step2_out/events/*_events.parquet`
  - `step2_out/meta/step2_config.json`

3. `step2_5_analyze_events.py`
- Purpose: Step 2.5 diagnostics, charts, compact summary JSON.
- Input: Step 2 events file.
- Output:
  - `step2_out/step2_5/step2_5_summary.json`
  - `step2_out/step2_5/figures/*.png`

4. `step2b_knob_sweep_backtest.py`
- Purpose: random knob sweep with 80/20 chronological split, no-ML vs ML-sim.
- Produces per-run summary and best trial equity plot.
- Usually called by `step2_compare_and_select.py`.

5. `step2_compare_and_select.py`
- Purpose: run multiple step2b scenarios (QQQ/SPY, cross on/off), compare them, save best candidates.
- Output:
  - `step2_out/selection/comparison_report.json`
  - `step2_out/selection/best_candidate_non_ml.json`
  - `step2_out/selection/best_candidate_ml.json`
  - per-scenario folders with `step2b_summary.json` and `best_equity_compare.png`

6. `step2_capital_curve.py`
- Purpose: load saved best candidates and plot no-ML vs ML candidate capital curve over history.
- Output:
  - `step2_out/selection/visual/capital_curve.png`
  - `step2_out/selection/visual/capital_curve_summary.json`

7. `step2_dual_symbol_portfolio_test.py`
- Purpose: choose best SPY + QQQ candidates and optimize portfolio weight into one single equity line.
- Output (per mode):
  - `step2_out/selection/dual_portfolio/*` (`no_ml`)
  - `step2_out/selection/dual_portfolio_ml/*` (`ml_sim`)

### Data scripts (upstream)
8. `data/scripts/download_history_spy_qqq.py`
- Purpose: pull IBKR hourly historical bars (SPY/QQQ).
- Output: raw files in `data/`.

9. `data_clean/scripts/clean_ibkr_prepped.py`
- Purpose: clean and session-validate prepped bars.
- Output: cleaned bars in `data_clean/`.

10. `data/scripts/visual.py`
- Purpose: ad-hoc visualization helper for raw data.

---

## `step2_out` Layout

### `step2_out/bar_features/`
- `qqq_bar_features.parquet`
- `spy_bar_features.parquet`
- Bar-level engineered features used for event construction and analysis.

### `step2_out/events/`
- `qqq_events.parquet` (or trade symbol equivalent)
- Event-level dataset with labels and costs (net-aware).

### `step2_out/meta/`
- `step2_config.json`
- Exact knobs/settings used for latest Step 2 build.

### `step2_out/step2_5/`
- `step2_5_summary.json`: compact analytics summary.
- `figures/`:
  - `equity_gross_vs_net.png`
  - `return_hist_bps.png`
  - `yearly_net_and_winrate.png`
  - `cost_sensitivity.png`
  - `decile_profiles.png`

### `step2_out/selection/`
- `comparison_report.json`
  - scenario-by-scenario comparison and best-candidate blocks.
- `best_candidate_non_ml.json`
  - selected no-ML candidate configuration + snapshot metrics.
- `best_candidate_ml.json`
  - selected ML-sim candidate configuration + snapshot metrics.

Subfolders:
- `qqq_cross_off/`, `qqq_cross_on/`, `spy_cross_off/`, `spy_cross_on/`
  - each contains:
    - `step2b_summary.json`
    - `best_equity_compare.png`
- `visual/`
  - `capital_curve.png`
  - `capital_curve_summary.json`
- `dual_portfolio/`
  - no-ML optimized SPY+QQQ single-line portfolio:
    - `dual_symbol_portfolio_curve.png`
    - `dual_symbol_portfolio_summary.json`
- `dual_portfolio_ml/`
  - ML-sim optimized SPY+QQQ single-line portfolio:
    - `dual_symbol_portfolio_curve.png`
    - `dual_symbol_portfolio_summary.json`

---

## Script-to-Output Map

1. `step2_build_events_dataset.py`
- writes `bar_features`, `events`, `meta`.

2. `step2_5_analyze_events.py`
- writes `step2_5/`.

3. `step2_compare_and_select.py`
- writes `selection/` scenario summaries + best candidate JSONs.

4. `step2_capital_curve.py`
- writes `selection/visual/`.

5. `step2_dual_symbol_portfolio_test.py`
- writes `selection/dual_portfolio/` or `selection/dual_portfolio_ml/`.

6. `step2_run_all.py`
- runs all of the above in sequence.

---

## Recommended Usage

To regenerate the whole Step 2 stack from cleaned data:

```powershell
python step2_run_all.py
```

This is the default “single command” workflow.
