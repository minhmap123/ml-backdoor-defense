# Backdoor Detection on Tabular IDS Models

Run the full IoTID20 benchmark:

```bash
python scripts/run_iotid20_full_benchmark.py
```

Summarize the latest benchmark run for any IDS benchmark with the same artifact layout:

```bash
python scripts/summarize_benchmark.py
```

The report is written to `results/benchmark_runs/<run>/reports/benchmark_report.md`, with CSV tables for training metrics and detection metrics in the same folder.
