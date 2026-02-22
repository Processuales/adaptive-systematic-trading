Final Output Layout

charts/: human-friendly summary charts for SPY+QQQ and Step 4 combined allocators
step2/: Step 2 raw artifacts (summary JSON + raw strategy chart)
step3/: Step 3 raw artifacts from optimizer export
dual ml/: optional pattern-aid ML comparison report
reports/: consolidated machine-readable summaries
combined/: SPYQQQ + BTCETH combined artifacts
  - charts/: combined equity curves and allocator curves
  - reports/: combined run summaries
  - data/: combined monthly tables

Automation entry points:
- reports/final_output_summary.json (SPY+QQQ)
- reports/step4_summary.json (combined absolute best)
- reports/step4_summary_dynamic.json (combined dynamic best)
