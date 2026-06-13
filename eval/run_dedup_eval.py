"""Evaluate ML predictions with downstream deduplication.

Strategy (high-recall + post-process for precision):
  1. Possession detector enabled with intermediate-loss → many loss frames
  2. ML predicts pass/cross/shot/none on each
  3. Dedup: cluster nearby same-label same-player predictions, keep highest prob
  4. Evaluate with same protocol as run_ml_eval.py

Hyperparam:
  - DEDUP_WINDOW: time window (s) within which preds are considered duplicates
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
from run_raw_eval import gt_extend
import run_raw_eval

ML_DIR = ROOT / "data" / "dfl" / "ml"
RAW_GT = ROOT / "data" / "dfl" / "processed" / "raw_gt_synced"
RAW_GT_SYNCED = ROOT / "data" / "dfl" / "processed" / "raw_gt_synced"
OUT = ROOT / "evaluation_results"

# Toggle: use sync-corrected GT timestamps if available
USE_SYNCED_GT = True
# Toggle: use Hungarian (optimal) 1:1 matching instead of greedy
USE_HUNGARIAN = False

DEDUP_WINDOW_DEFAULT = 1.0  # seconds


def dedup_predictions_v2(df: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Possession-aware dedup.

    Two same-label same-player preds within window_s are deduped ONLY IF
    no other player has a non-none prediction between them.
    This preserves A→B→A patterns (one-touch, give-and-go) where two A
    predictions are real separate events because B touched the ball in between.

    Algorithm:
      For each (match, period), iterate rows in time order. For each
      candidate i with pred_label=L:
        - Walk forward j > i within window_s
        - If row at t∈(t_i, t_j) has different player AND pred_label != "none":
            → barrier: stop dedup-comparison for i (don't merge across)
        - Else if same player AND same label: compare probs, drop lower one
    """
    df = df.sort_values(["match_id", "period_id", "timestamp"]).reset_index(drop=True)
    keep_mask = pd.Series(True, index=df.index)

    for (mid, pid), grp in df.groupby(["match_id", "period_id"]):
        grp = grp.sort_values("timestamp")
        idx_list = grp.index.tolist()
        n = len(idx_list)
        ts = grp["timestamp"].values
        players = grp["loss_player"].values
        labels = grp["pred_label"].values

        for i_pos in range(n):
            i = idx_list[i_pos]
            if not keep_mask[i]:
                continue
            lbl_i = labels[i_pos]
            if lbl_i not in ("pass", "cross", "shot"):
                continue
            prob_col = f"prob_{lbl_i}"
            player_i = players[i_pos]
            t_i = ts[i_pos]

            # Walk forward within window, tracking whether a barrier
            # (non-none prediction by different player) has appeared.
            barrier_seen = False
            for j_pos in range(i_pos + 1, n):
                j = idx_list[j_pos]
                t_j = ts[j_pos]
                if t_j - t_i > window_s:
                    break
                if not keep_mask[j]:
                    continue
                player_j = players[j_pos]
                lbl_j = labels[j_pos]

                # Different player AND a real event → barrier
                if player_j != player_i and lbl_j != "none":
                    barrier_seen = True
                    continue

                # Past a barrier: don't merge, but keep walking (others ignored)
                if barrier_seen:
                    continue

                # Same player + same label: real duplicate candidate
                if player_j == player_i and lbl_j == lbl_i:
                    p_i = grp.loc[i, prob_col]
                    p_j = grp.loc[j, prob_col]
                    if p_i >= p_j:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False
                        break  # i is dropped, no point continuing for i

    return df[keep_mask].copy()


def dedup_predictions(df: pd.DataFrame, window_s: float,
                      same_player_only: bool = True) -> pd.DataFrame:
    """Remove duplicate predictions within window_s, keep highest probability.

    For each label class, group nearby predictions:
      - If two preds of same label & (optionally) same player are within window_s
      - Keep the one with higher prob_<label>

    Returns deduplicated DataFrame.
    """
    keep_mask = pd.Series(True, index=df.index)
    df = df.sort_values(["match_id", "period_id", "timestamp"]).reset_index(drop=True)

    for label in ["pass", "cross", "shot"]:
        prob_col = f"prob_{label}"
        sub = df[df["pred_label"] == label].copy()
        # Build key for grouping: match + period (+ player optional)
        for (mid, pid), grp in sub.groupby(["match_id", "period_id"]):
            grp = grp.sort_values("timestamp")
            indices = grp.index.tolist()
            if same_player_only:
                # Group by player too
                for player, pgrp in grp.groupby("loss_player"):
                    pgrp = pgrp.sort_values("timestamp")
                    pidx = pgrp.index.tolist()
                    # Greedy walk: if neighbor within window, keep higher prob
                    for i in range(len(pidx)):
                        if not keep_mask[pidx[i]]:
                            continue
                        for j in range(i + 1, len(pidx)):
                            if not keep_mask[pidx[j]]:
                                continue
                            dt = (pgrp.loc[pidx[j], "timestamp"]
                                  - pgrp.loc[pidx[i], "timestamp"])
                            if dt > window_s:
                                break
                            # Compare probs
                            if pgrp.loc[pidx[i], prob_col] >= pgrp.loc[pidx[j], prob_col]:
                                keep_mask[pidx[j]] = False
                            else:
                                keep_mask[pidx[i]] = False
                                break
            else:
                # Same as above without player grouping
                pidx = indices
                for i in range(len(pidx)):
                    if not keep_mask[pidx[i]]:
                        continue
                    for j in range(i + 1, len(pidx)):
                        if not keep_mask[pidx[j]]:
                            continue
                        dt = (grp.loc[pidx[j], "timestamp"]
                              - grp.loc[pidx[i], "timestamp"])
                        if dt > window_s:
                            break
                        if grp.loc[pidx[i], prob_col] >= grp.loc[pidx[j], prob_col]:
                            keep_mask[pidx[j]] = False
                        else:
                            keep_mask[pidx[i]] = False
                            break
    return df[keep_mask].copy()


def run_eval(ml: pd.DataFrame, tag: str = ""):
    label_map = {"none": pd.NA, "pass": "Pass",
                 "cross": "Cross", "shot": "ShotOffTarget"}
    ml = ml.copy()
    ml["event_name"] = ml["pred_label"].map(label_map)
    ml["event_player"] = ml["loss_player"]
    ml["event_team"] = ml["loss_team"]

    rows = []
    for mid in sorted(ml["match_id"].unique()):
        s = ml[ml["match_id"] == mid]
        pred_open = prepare_pred_open(s)
        synced_path = RAW_GT_SYNCED / f"{mid}.parquet"
        if USE_SYNCED_GT and synced_path.exists():
            raw = pd.read_parquet(synced_path)
            # Replace timestamp with sync_ts before passing to gt_extend
            raw = raw.copy()
            raw["timestamp"] = raw["sync_ts"]
        else:
            raw = pd.read_parquet(RAW_GT / f"{mid}.parquet")
        gt = gt_extend(raw)
        for lbl in OPEN_LABELS:
            r = evaluate_paper(gt, pred_open, lbl,
                               window=PAPER_WINDOW, use_player=False,
                               use_hungarian=USE_HUNGARIAN)
            r["match"] = mid
            rows.append(pd.Series(r))

    df = pd.DataFrame(rows)
    g = df.groupby("label")[["GT", "Pred", "TP", "FP", "FN"]].sum()
    g["Precision"] = (g["TP"] / (g["TP"] + g["FP"])).round(3)
    g["Recall"] = (g["TP"] / (g["TP"] + g["FN"])).round(3)
    d = (g["Precision"] + g["Recall"]).replace(0, 1)
    g["F1"] = (2 * g["Precision"] * g["Recall"] / d).round(3)
    tp, fp, fn = int(g["TP"].sum()), int(g["FP"].sum()), int(g["FN"].sum())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"--- {tag} ---")
    print(g.to_string())
    print(f"Micro: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    return g, (p, r, f1)


def main():
    ml = pd.read_parquet(ML_DIR / "xgb_predictions.parquet")
    print(f"Total ML preds: {len(ml)}")
    print(f"  baseline label dist: {ml['pred_label'].value_counts().to_dict()}")
    print()

    # No dedup baseline
    run_eval(ml, "Baseline (no dedup)")

    print()
    # Sweep dedup windows: compare v1 (same-player) vs v2 (possession-aware)
    summary_rows = []
    for win in [2.0, 3.0, 4.0, 5.0]:
        for tag, fn in [
            ("v1_same_player", lambda d, w=win: dedup_predictions(d, w, same_player_only=True)),
            ("v2_possession",  lambda d, w=win: dedup_predictions_v2(d, w)),
        ]:
            print(f"\n=== {tag} win={win}s ===")
            deduped = fn(ml)
            print(f"  After dedup: {len(deduped)} (-{len(ml) - len(deduped)})")
            print(f"  Label dist: {deduped['pred_label'].value_counts().to_dict()}")
            g, (p, r, f1) = run_eval(deduped, f"{tag} {win}s")
            for lbl, row in g.iterrows():
                summary_rows.append({
                    "variant": tag, "win": win, "label": lbl,
                    "P": row["Precision"], "R": row["Recall"], "F1": row["F1"],
                })
            summary_rows.append({
                "variant": tag, "win": win, "label": "MICRO",
                "P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
            })

    print("\n" + "=" * 60)
    print("Summary (all 7 matches):")
    summary = pd.DataFrame(summary_rows)
    pivot = summary.pivot_table(index=["variant", "win"], columns="label",
                                values=["P", "R", "F1"])
    print(pivot.to_string())


if __name__ == "__main__":
    main()
