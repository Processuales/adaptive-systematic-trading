# Side Pairs

This folder is for pair experiments that are **not** the main production benchmark.

Current example:
- `ibit_etha/`

Recommended structure for a new pair (example: `btc_eth/`):

- `data/raw/`
- `scripts/`
- `final output/charts/`
- `final output/reports/final_output_summary.json`

Keep heavy generated folders local only:
- `step2_out/`
- `step3_out/`
- `data_clean/`
- `data_clean_alias/`

The main benchmark stays at repository root (`SPY + QQQ`).
