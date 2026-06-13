"""Compare ML vs Rule-based using identical evaluation pipeline.

Converts ML predictions back to open_result.parquet-like format,
then evaluates using the same matching as run_raw_eval.py.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "elastic_nw"))
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from models.AutoEvent.evaluate import (
    evaluate_paper, prepare_pred_open, OPEN_LABELS, PAPER_WINDOW,
)
from run_raw_eval import gt_extend, _load_person_map
import run_raw_eval

ML_DIR = ROOT / "data" / "dfl" / "ml"
ELASTIC = ROOT / "data" / "dfl" / "elastic"
OUT = ROOT / "evaluation_results"


def ml_to_open_result(ml_pred: pd.DataFrame) -> pd.DataFrame:
    """Convert ML predictions to a DataFrame compatible with prepare_pred_open."""
    df = ml_pred.copy()
    # Map ML pred_label → open_result event_name
    label_map = {
        "none": pd.NA,
        "pass": "Pass",
        "cross": "Cross",
        "shot": "ShotOffTarget",  # default; could refine using probabilities
    }
    df["event_name"] = df["pred_label"].map(label_map)
    df["event_player"] = df["loss_player"]
    df["event_team"] = df["loss_team"]
    return df


def main():
    # Load LOOCV predictions
    ml = pd.read_parquet(ML_DIR / "xgb_predictions.parquet")
    print(f"ML predictions loaded: {len(ml)} samples")
    print(f"  pred_label dist: {ml['pred_label'].value_counts().to_dict()}")

    matches = sorted(ml["match_id"].unique())

    # Build per-match ml-style "open_result" and run eval
    all_rows = []
    for mid in matches:
        ml_mid = ml[ml["match_id"] == mid].copy()
        pred_df = ml_to_open_result(ml_mid)
        # Apply prepare_pred_open
        pred_open = prepare_pred_open(pred_df)

        # Build EXTEND GT
        run_raw_eval._person_to_player_cache = _load_person_map(mid)
        raw = pd.read_parquet(ROOT / "data" / "dfl" / "raw_gt" / f"{mid}.parquet")
        gt = gt_extend(raw)

        for lbl in OPEN_LABELS:
            r = evaluate_paper(gt, pred_open, lbl,
                               window=PAPER_WINDOW, use_player=False)
            r["match"] = mid
            all_rows.append(pd.Series(r))

    df = pd.DataFrame(all_rows)
    g = df.groupby("label")[["GT", "Pred", "TP", "FP", "FN"]].sum()
    g["Precision"] = (g["TP"] / (g["TP"] + g["FP"])).round(3)
    g["Recall"] = (g["TP"] / (g["TP"] + g["FN"])).round(3)
    d = (g["Precision"] + g["Recall"]).replace(0, 1)
    g["F1"] = (2 * g["Precision"] * g["Recall"] / d).round(3)
    tp, fp, fn = int(g["TP"].sum()), int(g["FP"].sum()), int(g["FN"].sum())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print()
    print("=" * 60)
    print("ML (LOOCV) — using run_raw_eval matching (raw GT EXTEND, "
          "±10s, use_player=False)")
    print("=" * 60)
    print(g.to_string())
    print(f"Micro: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    micro = pd.DataFrame([{
        "GT": int(g["GT"].sum()), "Pred": int(g["Pred"].sum()),
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3),
    }], index=["MICRO"])
    out_df = pd.concat([g, micro])
    out_df.to_csv(OUT / "open_FINAL_ml_loocv.csv")
    print(f"\nSaved to {OUT / 'open_FINAL_ml_loocv.csv'}")


if __name__ == "__main__":
    main()
