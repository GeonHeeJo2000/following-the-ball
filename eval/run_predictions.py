"""Convert rule-based AutoEvent output to loss-frame prediction parquet.

Maps detected events (Pass / Cross / Shot{OnTarget,OffTarget} / Save)
to per-loss-frame predictions in the same schema as XGBoost
predictions, so they can be compared in the same eval pipeline

For each loss frame in a match's possession.parquet:
  - Find rule-based events with same period within ±2s and same player.
  - If any such event maps to Pass/Cross/Shot → pred_label = that type.
  - Else → pred_label = none.

Saves to data/dfl/ml/rule_predictions.parquet (test-match subset
mirroring xgb_predictions.parquet).
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

ML_DIR = ROOT / "data" / "dfl" / "ml"

LABELS = ["none", "pass", "cross", "shot"]
WINDOW_S = 2.0

EVENT_MAP = {
    "Pass": "pass",
    "Cross": "cross",
    "ShotOnTarget": "shot",
    "ShotOffTarget": "shot",
    "Save": "shot",
}


def reproduce_test():
    info_path = ML_DIR / "kick" / "info.txt"
    test_match_ids: list[str] = []
    with open(info_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("test_match_ids:"):
                test_match_ids = eval(line.split(":", 1)[1].strip())
                break
    return sorted(test_match_ids)


def predict_for_match(mid):
    # Loss frames + GT label (from feature parquet — already aligned)
    feat_path = ML_DIR / "kick" / f"{mid}.parquet"
    if not feat_path.exists():
        print(f"  [skip] no feature parquet {mid}")
        return None
    feat = pd.read_parquet(feat_path)

    # Rule-based events
    op_path = ML_DIR / "detection" / mid / "open_play.parquet"
    if not op_path.exists():
        print(f"  [skip] no rule-based output {mid}")
        return None
    op = pd.read_parquet(op_path)
    print(f"  {mid}: {len(feat)} loss frames")

    # Index rule events by (period, frame_id). Each loss frame in feat
    # corresponds 1-1 to a loss frame in possession.parquet; rule-based
    # AutoEvent emits its event at the SAME frame_id. So we look up by
    # exact (period, frame_id) — no ±s window needed.
    events_with_fid = op[op.event_name.notna()][
        ["period_id", "frame_id", "event_name", "event_player"]]
    events_with_fid = events_with_fid[
        events_with_fid.event_name.isin(EVENT_MAP)]
    ev_lookup = {(int(p), int(f)): k
                 for p, f, k in zip(events_with_fid.period_id,
                                    events_with_fid.frame_id,
                                    events_with_fid.event_name)}

    rows = []
    for _, r in feat.iterrows():
        key = (int(r.period_id), int(r.frame_id))
        kind = ev_lookup.get(key)
        rows.append(EVENT_MAP.get(kind, "none") if kind else "none")

    feat = feat.copy()
    feat["pred_label"] = rows
    for lab in LABELS:
        feat[f"prob_{lab}"] = (feat["pred_label"] == lab).astype(float)
    return feat


def main():
    test_mids = reproduce_test()
    print(f"Generating rule-based predictions for {len(test_mids)} matches")

    all_dfs = []
    for mid in test_mids:
        out = predict_for_match(mid)
        if out is not None:
            all_dfs.append(out)

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = ML_DIR / "predictions" / "rule_predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    print(f"\nSaved {len(combined)} rows → {out_path}")

    # Per-frame F1 summary
    from sklearn.metrics import classification_report
    print("\nPer-frame F1:")
    print(classification_report(combined.label, combined.pred_label,
                                labels=LABELS, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
