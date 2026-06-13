"""Evaluate model predictions against raw DFL XML GT (bypassing elastic).

Uses data/dfl/raw_gt_synced/<match>.parquet produced by build_raw_gt_synced.py.

Provides several GT mapping presets to compare:
  - STRICT  : DFL Pass→pass, Cross→cross, ShotAtGoal→shot
  - EXTEND  : + OtherBallAction→pass, TacklingGame[Loser side]→pass
  - MAXIMAL : EXTEND + BallClaiming→claim (separate), Foul→pass
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "elastic"))
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from models.AutoEvent.evaluate import (
    evaluate_paper as ae_evaluate_paper,
    prepare_pred_open as ae_prepare_pred_open,
    confusion_matrix_open as ae_cm_open,
    OPEN_LABELS, PAPER_WINDOW,
)

ELASTIC_DIR = ROOT / "data" / "dfl" / "elastic"
RAW_GT_DIR = ROOT / "data" / "dfl" / "raw_gt_synced"
OUT = ROOT / "evaluation_results"


def gt_strict(raw_gt: pd.DataFrame) -> pd.DataFrame:
    """STRICT: pass / cross / shot only — direct DFL labels."""
    rows = []
    for _, r in raw_gt.iterrows():
        kind = r["event_kind"]
        if kind == "Pass":
            label = "pass"
        elif kind == "Cross":
            label = "cross"
        elif kind == "ShotAtGoal":
            label = "shot"
        else:
            continue
        if not isinstance(r["player_id"], str):
            continue
        rows.append({
            "period_id": int(r["period_id"]),
            "timestamp": float(r["timestamp"]),
            "object_id": r["player_id"],
            "label": label,
        })
    return pd.DataFrame(rows)


def gt_extend(raw_gt: pd.DataFrame) -> pd.DataFrame:
    """EXTEND: STRICT + OBA[defensive_clearance] + TacklingGame[Loser-control].

    Only DEFENSIVE_CLEARANCE OBAs are included as 'pass' (intentional
    action). Other OBAs (loose touch, bad control) are excluded — they
    don't have ML predictions and would create matching cascades.
    """
    rows = []
    for _, r in raw_gt.iterrows():
        kind = r["event_kind"]
        label = None
        actor = r["player_id"]

        if kind == "Pass":
            label = "pass"
        elif kind == "Cross":
            label = "cross"
        elif kind == "ShotAtGoal":
            label = "shot"
        elif kind == "OtherBallAction":
            if str(r.get("defensive_clearance", "")).lower() == "true":
                label = "pass"
            else:
                continue
        # TacklingGame excluded — consistent with new labelers.

        if label is None or not isinstance(actor, str):
            continue
        # Prefer elastic-synced timestamp when present.
        ev_ts = r.get("sync_ts", r["timestamp"])
        if not np.isfinite(ev_ts):
            ev_ts = r["timestamp"]
        rows.append({
            "period_id": int(r["period_id"]),
            "timestamp": float(ev_ts),
            "object_id": actor,
            "label": label,
        })
    return pd.DataFrame(rows)


def gt_maximal(raw_gt: pd.DataFrame) -> pd.DataFrame:
    """MAXIMAL: EXTEND + Foul→pass + ShotWide/SavedShot/SuccessfulShot/BlockedShot→shot."""
    rows = []
    for _, r in raw_gt.iterrows():
        kind = r["event_kind"]
        label = None
        actor = r["player_id"]

        if kind == "Pass" or kind == "OtherBallAction" or kind == "Foul":
            label = "pass"
        elif kind == "Cross":
            label = "cross"
        elif kind in ("ShotAtGoal", "ShotWide", "SavedShot",
                       "SuccessfulShot", "BlockedShot"):
            label = "shot"
        elif kind == "TacklingGame":
            if r.get("loser_role") == "withBallControl":
                label = "pass"
                actor = _person_to_player_cache.get(r["loser_id"], actor)

        if label is None or not isinstance(actor, str):
            continue
        rows.append({
            "period_id": int(r["period_id"]),
            "timestamp": float(r["timestamp"]),
            "object_id": actor,
            "label": label,
        })
    return pd.DataFrame(rows)


# Global cache of DFL-OBJ-XXX → home_X / away_X for the current match
_person_to_player_cache: dict[str, str] = {}


def _load_person_map(match_id: str) -> dict[str, str]:
    """Reuse raw_dfl_gt's player map by re-parsing the meta XML for this match."""
    import xml.etree.ElementTree as ET
    from build.build_raw_gt_synced import build_player_map, RAW_DIR
    match_dir = RAW_DIR / match_id
    # Support both naming conventions
    cands = (list(match_dir.glob("DFL_02_01_matchinformation*.xml"))
             + list(match_dir.glob("DFL-02.01-Spielinformationen*.xml")))
    if not cands:
        # Fallback: use elastic teams.parquet (XML may have been deleted)
        teams_path = (ROOT / "data" / "dfl" / "elastic"
                      / match_id / "teams.parquet")
        if teams_path.exists():
            df = pd.read_parquet(teams_path)
            # person_id (player_id from XML) → object_id (home_X/away_X)
            return dict(zip(df["player_id"], df["object_id"]))
        raise FileNotFoundError(
            f"No matchinformation XML or teams.parquet for {match_id}")
    meta_xml = cands[0]
    return build_player_map(ET.parse(meta_xml).getroot())


def aggregate(rows_list):
    df = pd.concat(rows_list)
    g = df.groupby("label")[["GT", "Pred", "TP", "FP", "FN"]].sum()
    g["Precision"] = (g["TP"] / (g["TP"] + g["FP"])).round(3)
    g["Recall"] = (g["TP"] / (g["TP"] + g["FN"])).round(3)
    d = (g["Precision"] + g["Recall"]).replace(0, 1)
    g["F1"] = (2 * g["Precision"] * g["Recall"] / d).round(3)
    return g


def micro(g):
    tp = int(g["TP"].sum()); fp = int(g["FP"].sum()); fn = int(g["FN"].sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(TP=tp, FP=fp, FN=fn,
                Precision=round(p, 3), Recall=round(r, 3), F1=round(f1, 3))


def main():
    global _person_to_player_cache
    matches = sorted([d.name for d in ELASTIC_DIR.iterdir()
                      if (d / "tracking.parquet").exists()])

    OUT.mkdir(exist_ok=True)
    presets = [
        ("STRICT",  gt_strict),
        ("EXTEND",  gt_extend),
        ("MAXIMAL", gt_maximal),
    ]

    for use_player in [True, False]:
        for preset_name, gt_fn in presets:
            rows = []
            cm_total = None
            for mid in matches:
                _person_to_player_cache = _load_person_map(mid)
                raw_gt = pd.read_parquet(RAW_GT_DIR / f"{mid}.parquet")
                gt = gt_fn(raw_gt)
                open_res = pd.read_parquet(
                    ELASTIC_DIR / mid / "autoevent_cache" / "open_result.parquet"
                )
                pred = ae_prepare_pred_open(open_res)
                for lbl in OPEN_LABELS:
                    r = ae_evaluate_paper(gt, pred, lbl,
                                          window=PAPER_WINDOW,
                                          use_player=use_player)
                    r["match"] = mid
                    rows.append(pd.Series(r))
                cm = ae_cm_open(gt, pred, window=PAPER_WINDOW,
                                use_player=use_player)
                cm_total = cm if cm_total is None else cm_total.add(cm, fill_value=0)
            df = pd.DataFrame(rows)
            g = aggregate([df])
            m = micro(g)
            tag = f"raw_{preset_name.lower()}_player_{'true' if use_player else 'false'}"
            print(f"--- {preset_name} GT, use_player={use_player} ---")
            print(g.to_string())
            print(f"  Micro: {m}")
            print()

            # Save
            micro_row = pd.DataFrame([{
                "GT": int(g["GT"].sum()), "Pred": int(g["Pred"].sum()),
                **m,
            }], index=["MICRO"])
            out_df = pd.concat([g, micro_row])
            out_df.to_csv(OUT / f"open_{tag}.csv")
            cm_total.fillna(0).astype(int).to_csv(OUT / f"open_{tag}_confusion.csv")


if __name__ == "__main__":
    main()
