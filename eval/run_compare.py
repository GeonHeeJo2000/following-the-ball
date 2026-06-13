"""Compare Rule-based / ML model predictions on the same test set.

Modes:
  --quick:   single (τ, win) config, greedy only — fast comparison
             (~1-2 min for all available models)
  default:   τ × window × greedy/Hungarian sweep, prints top-3 per
             method per model (~10-15 min for 4 models)

Models picked up automatically if their prediction parquet exists in
``data/dfl/ml/``:
  - XGBoost (filtered to test subset for fair compare)
"""
from __future__ import annotations
import argparse
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

import run_dedup_eval
from run_dedup_eval import dedup_predictions_v2, run_eval

ML_DIR = ROOT / "data" / "dfl" / "ml"

def apply_tau_override(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Override pred_label='none' → 'pass' when prob_pass ≥ τ.

    Cross/shot argmax preserved. Only flips none→pass.
    τ=0 disables (no override).
    """
    if tau <= 0:
        return df
    df = df.copy()
    flip_mask = (df["pred_label"] == "none") & (df["prob_pass"] >= tau)
    df.loc[flip_mask, "pred_label"] = "pass"
    return df

def reproduce_test_matches(train_ratio=0.8, valid_ratio=0.1):
    """Sequential 80/10/10 split — same logic as dataset.split_match_ids().

    match_ids are taken from the elastic directory (same source as Dataset).
    """
    import os
    elastic_path = ROOT / "data" / "dfl" / "processed" / "elastic"
    matches = sorted([
        d for d in os.listdir(elastic_path)
        if (elastic_path / d / "tracking.parquet").exists()
    ])
    train_end = int(len(matches) * train_ratio)
    valid_end = int(len(matches) * (train_ratio + valid_ratio))
    return matches[valid_end:]


def quick_eval(name: str, df: pd.DataFrame, tau: float, win: float,
               hungarian: bool = False):
    """One-shot eval at a single config — for fast comparison."""
    run_dedup_eval.USE_HUNGARIAN = hungarian
    ml = apply_tau_override(df, tau)
    if win > 0:
        ml = dedup_predictions_v2(ml, win)
    g, (p, r, f1) = run_eval(ml, f"{name} τ={tau} w={win}")
    method = "Hungarian" if hungarian else "Greedy"
    print(f"\n[{name} | {method} | τ={tau} w={win}]")
    print(f"  Pass:  P={g.loc['pass']['Precision']:.3f} "
          f"R={g.loc['pass']['Recall']:.3f} "
          f"F1={g.loc['pass']['F1']:.3f}")
    print(f"  Cross: F1={g.loc['cross']['F1']:.3f}")
    print(f"  Shot:  F1={g.loc['shot']['F1']:.3f}")
    print(f"  Micro: F1={f1:.3f}", flush=True)


def sweep_eval(name: str, df: pd.DataFrame, taus, wins,
               include_hungarian=True):
    """τ × window sweep with both greedy and Hungarian, print top-3."""
    print(f"\n{'='*65}")
    print(f"{name}  ({len(df)} preds, "
          f"{df['match_id'].nunique()} matches)")
    print(f"{'='*65}")
    rows = []
    for tau in taus:
        for win in wins:
            ml = apply_tau_override(df, tau)
            if win > 0:
                ml = dedup_predictions_v2(ml, win)
            for hungarian in ([False, True] if include_hungarian else [False]):
                run_dedup_eval.USE_HUNGARIAN = hungarian
                g, (p, r, f1) = run_eval(ml, f"{tau}/{win}")
                rows.append({
                    "tau": tau, "win": win,
                    "method": "hungarian" if hungarian else "greedy",
                    "pass_P": g.loc["pass"]["Precision"],
                    "pass_R": g.loc["pass"]["Recall"],
                    "pass_F1": g.loc["pass"]["F1"],
                    "cross_F1": g.loc["cross"]["F1"],
                    "shot_F1": g.loc["shot"]["F1"],
                    "micro_F1": round(f1, 3),
                })
    s = pd.DataFrame(rows)
    for method in (["greedy"] + (["hungarian"] if include_hungarian else [])):
        sub = s[s["method"] == method].sort_values("pass_F1",
                                                   ascending=False).head(3)
        print(f"\n[{method}]")
        print(sub.to_string(index=False))
    return s


def _iter_models(test_matches):
    """Yield (name, df) for every available prediction parquet.

    Always intersects predictions with ``test_matches`` so every model is
    scored on the *same* set of matches, regardless of which test split
    that model was originally trained against.
    """
    candidates = [
        ("Rule-based v3",
         ML_DIR / "rule_predictions.parquet"),
        ("XGBoost (sync_ts)",
         ML_DIR / "predictions" / "xgb_predictions.parquet"),
    ]
    for name, path in candidates:
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["match_id"].isin(test_matches)].reset_index(drop=True)
        if df.empty:
            print(f"  [skip] {name}: no overlap with test_matches",
                  flush=True)
            continue
        yield name, df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="single (τ, win) greedy config — fast (~1-2 min)")
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--win", type=float, default=2.0)
    args = parser.parse_args()

    test_matches = reproduce_test_matches()
    print(f"Test matches ({len(test_matches)}): {test_matches}")

    if args.quick:
        for name, df in _iter_models(test_matches):
            quick_eval(name, df, args.tau, args.win, hungarian=False)
    else:
        taus = [0.20, 0.30, 0.40]
        wins = [0.0, 1.0, 2.0, 3.0]
        for name, df in _iter_models(test_matches):
            sweep_eval(name, df, taus, wins, include_hungarian=True)


if __name__ == "__main__":
    main()
