#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.iotid20 import IoTID20Dataset
from src.data.cic_ids2017 import CICIDS2017Dataset
from src.data.cic_ids2018 import CSECICIDS2018Dataset
from src.data.cic_iot_2023 import CICIoT2023Dataset

OUT_DIR = ROOT_DIR / "scripts" / "trigger_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("iotid20", IoTID20Dataset, "Normal"),
    ("cic_ids2017", CICIDS2017Dataset, "Benign"),
    ("cic_ids2018", CSECICIDS2018Dataset, "Benign"),
    ("cic_iot_2023", CICIoT2023Dataset, "Benign"),
]

for name, cls, benign_label_str in DATASETS:
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    ds = cls()

    prepared = ds.prepare_clean_partitions()
    x_raw = prepared["x_train_raw"]
    y_enc = prepared["y_train"].to_numpy()
    encoder = prepared["encoder"]
    feat_names = prepared["feature_names"]

    classes = list(encoder.classes_)
    benign_idx = classes.index(benign_label_str)
    benign_mask = (y_enc == benign_idx)
    x_benign = x_raw.iloc[benign_mask]

    print(f"  classes: {classes}")
    print(f"  feature count: {len(feat_names)}")
    print(f"  total samples: {len(x_raw):,}, benign: {len(x_benign):,}", flush=True)

    stats = []
    for i, col in enumerate(feat_names):
        s = x_benign[col].to_numpy()
        s_all = x_raw[col].to_numpy()
        q = np.percentile(s, [0, 25, 50, 75, 95, 99, 100])
        q_all = np.percentile(s_all, [0, 50, 100])
        stats.append({
            "idx": i,
            "name": col,
            "benign": {
                "min": float(q[0]),
                "p25": float(q[1]),
                "p50": float(q[2]),
                "p75": float(q[3]),
                "p95": float(q[4]),
                "p99": float(q[5]),
                "max": float(q[6]),
                "unique_count": int(np.unique(s[~np.isnan(s)]).size),
            },
            "all": {
                "min": float(q_all[0]),
                "p50": float(q_all[1]),
                "max": float(q_all[2]),
            },
        })

    out = {
        "dataset": name,
        "classes": classes,
        "benign_label_idx": benign_idx,
        "feature_count": len(feat_names),
        "total_samples": int(len(x_raw)),
        "benign_samples": int(len(x_benign)),
        "feature_stats": stats,
    }
    out_path = OUT_DIR / f"{name}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path} in {time.time()-t0:.1f}s", flush=True)

print("\nDone.")
