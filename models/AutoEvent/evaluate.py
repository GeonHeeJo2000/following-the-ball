"""
Evaluation utilities for Automatic Event Detection.

Provides:
  - Label maps (GT_TYPE_MAP, SP_MAP_GT, PRED_TYPE_MAP)
  - evaluate()               : time-window greedy matching (set piece)
  - evaluate_paper()         : time-window + player greedy matching (open play)
  - aggregate()              : compute P/R/F1 from per-match rows
  - prepare_gt_sp()          : build set-piece GT DataFrame from a SportecData object
  - prepare_pred_sp()        : build set-piece pred DataFrame from pipeline output
  - prepare_pred_open()      : build open-play pred DataFrame from openplay output
  - confusion_matrix_sp()    : build confusion matrix (GT vs predicted set piece labels)
  - eval_sp_match()          : run set-piece evaluation for one match
  - eval_sp_all_matches()    : run eval_sp_match() across a list of match IDs
  - eval_match_rdp()         : open-play eval for one match (parquet GT)
  - eval_all_matches_rdp()   : open-play eval across all matches (parquet GT)
"""

from __future__ import annotations

import pickle

import pandas as pd
import numpy as np

from elastic.tools.match_data import MatchData
from elastic.tools.sportec_data import SportecData

# ── Label maps ────────────────────────────────────────────────────────────────

GT_TYPE_MAP: dict[str, str] = {
    "Pass":         "pass",
    "Cross":        "cross",
    "Shot":         "shot",
}

SP_MAP_GT: dict[str, str] = {
    "ThrowIn":    "throw_in",
    "GoalKick":   "goal_kick",
    "CornerKick": "corner_kick",
    "FreeKick":   "free_kick",
    "KickOff":    "kickoff",
    "Penalty":    "penalty_kick",
}

PRED_TYPE_MAP: dict[str, str] = {
    "pass":           "pass",
    "cross":          "cross",
    "shot_on_target": "shot",
    "shot_off_target":"shot",
    "interception":   "interception",
}

OPEN_LABELS: list[str] = ["pass", "cross", "shot"]
SP_LABELS:   list[str] = ["throw_in", "goal_kick", "corner_kick", "free_kick", "kickoff", "penalty_kick"]

# Default evaluation windows (seconds)
PAPER_WINDOW: float = 10.0   # open play: ±10s + player
SP_WINDOW:    float = 10.0   # set piece: ±10s


# ── Core evaluation functions ─────────────────────────────────────────────────

def evaluate(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label: str,
    window: float = SP_WINDOW,
) -> dict:
    """
    Time-window greedy 1:1 matching.

    Both DataFrames must have columns: period_id, timestamp, label.
    """
    gt_sub   = gt_df[gt_df["label"] == label][["period_id", "timestamp"]].copy().reset_index(drop=True)
    pred_sub = pred_df[pred_df["label"] == label][["period_id", "timestamp"]].copy().reset_index(drop=True)

    matched_gt   = set()
    matched_pred = set()

    for gi, gr in gt_sub.iterrows():
        candidates = pred_sub[
            (pred_sub["period_id"] == gr["period_id"]) &
            (abs(pred_sub["timestamp"] - gr["timestamp"]) <= window)
        ]
        candidates = candidates[~candidates.index.isin(matched_pred)]
        if candidates.empty:
            continue
        best_pi = (candidates["timestamp"] - gr["timestamp"]).abs().idxmin()
        matched_gt.add(gi)
        matched_pred.add(best_pi)

    tp = len(matched_gt)
    fp = len(pred_sub) - len(matched_pred)
    fn = len(gt_sub) - tp
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "label": label, "GT": len(gt_sub), "Pred": len(pred_sub),
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3),
    }


def evaluate_paper(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label: str,
    window: float = PAPER_WINDOW,
    use_player: bool = True,
    use_hungarian: bool = True,
) -> dict:
    """
    Time-window 1:1 matching with optional player constraint.

    Default is greedy (paper-style). Set use_hungarian=True to use
    optimal 1:1 assignment via Hungarian algorithm — recovers cases
    where greedy mis-pairs predictions with the wrong GT.

    gt_df   must have: period_id, timestamp, object_id, label
    pred_df must have: period_id, timestamp, event_player, label
    """
    gt_sub   = gt_df[gt_df["label"] == label][["period_id", "timestamp", "object_id"]].copy().reset_index(drop=True)
    pred_sub = pred_df[pred_df["label"] == label][["period_id", "timestamp", "event_player"]].copy().reset_index(drop=True)

    matched_gt   = set()
    matched_pred = set()

    if use_hungarian:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        # Per-period optimal assignment
        for period in gt_sub["period_id"].unique():
            gi_idx = gt_sub.index[gt_sub["period_id"] == period].tolist()
            pi_idx = pred_sub.index[pred_sub["period_id"] == period].tolist()
            if not gi_idx or not pi_idx:
                continue
            # Cost = |Δt|, infeasible (>window or player mismatch) → big cost
            INF = 1e6 # 1,000,000 seconds should be "infinite" for our purposes
            
            cost = np.full((len(gi_idx), len(pi_idx)), INF)
            for ii, gi in enumerate(gi_idx):
                gt = gt_sub.at[gi, "timestamp"]
                gp = gt_sub.at[gi, "object_id"]
                for jj, pi in enumerate(pi_idx):
                    pt = pred_sub.at[pi, "timestamp"]
                    if abs(pt - gt) > window:
                        continue
                    if use_player and pred_sub.at[pi, "event_player"] != gp:
                        continue
                    cost[ii, jj] = abs(pt - gt)
            row_idx, col_idx = linear_sum_assignment(cost)
            for ii, jj in zip(row_idx, col_idx):
                if cost[ii, jj] < INF:
                    matched_gt.add(gi_idx[ii])
                    matched_pred.add(pi_idx[jj])
                
    else:
        for gi, gr in gt_sub.iterrows():
            cand = pred_sub[
                (pred_sub["period_id"] == gr["period_id"]) &
                (abs(pred_sub["timestamp"] - gr["timestamp"]) <= window)
            ]
            if use_player:
                cand = cand[cand["event_player"] == gr["object_id"]]
            cand = cand[~cand.index.isin(matched_pred)]
            if cand.empty:
                continue
            best_pi = (cand["timestamp"] - gr["timestamp"]).abs().idxmin()
            matched_gt.add(gi)
            matched_pred.add(best_pi)

    tp = len(matched_gt)
    fp = len(pred_sub) - len(matched_pred)
    fn = len(gt_sub) - tp
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "label": label, "GT": len(gt_sub), "Pred": len(pred_sub),
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-match (or per-label) result rows into a summary DataFrame
    with micro-averaged P/R/F1 columns.
    """
    df = pd.DataFrame(rows)
    summary = df.groupby("label")[["GT", "Pred", "TP", "FP", "FN"]].sum()
    summary["Precision"] = (summary["TP"] / (summary["TP"] + summary["FP"])).round(3)
    summary["Recall"]    = (summary["TP"] / (summary["TP"] + summary["FN"])).round(3)
    denom = summary["Precision"] + summary["Recall"]
    summary["F1"] = (2 * summary["Precision"] * summary["Recall"] / denom.where(denom > 0, other=1)).round(3)
    return summary


def micro_summary(summary_df: pd.DataFrame) -> dict:
    """Return micro-averaged totals from an aggregate() result."""
    tp = summary_df["TP"].sum()
    fp = summary_df["FP"].sum()
    fn = summary_df["FN"].sum()
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3)}


# ── Data preparation helpers ──────────────────────────────────────────────────

def prepare_gt_sp(match: SportecData) -> pd.DataFrame:
    """
    Build set-piece GT DataFrame from a SportecData object.

    GT set piece events are rows where `set_piece_type` is not None.
    timestamp is computed as seconds from period start (same as tracking).

    Returns
    -------
    gt_sp : columns: period_id, timestamp, object_id, label
            label is the common set-piece label (SP_MAP_GT values)
    """
    events = SportecData.find_object_ids(match.lineup, match.events)
    events = MatchData.calculate_event_seconds(events)

    gt_sp = events[events["set_piece_type"].isin(SP_MAP_GT)].copy()
    gt_sp["label"] = gt_sp["set_piece_type"].map(SP_MAP_GT)
    return gt_sp[["period_id", "timestamp", "object_id", "label"]].reset_index(drop=True)


def prepare_pred_sp(result: pd.DataFrame) -> pd.DataFrame:
    """
    Build set-piece pred DataFrame from SetPieceDetector pipeline output.

    Parameters
    ----------
    result : output of SetPieceDetector.run()

    Returns
    -------
    pred_sp : columns: period_id, timestamp, trigger_player, label
              label matches SP_MAP_GT value space
    """
    # FreeKick? 레이블은 평가 시 free_kick으로 취급
    label_map = {
        "ThrowIn":    "throw_in",
        "GoalKick":   "goal_kick",
        "CornerKick": "corner_kick",
        "FreeKick":   "free_kick",
        "FreeKick?":  "free_kick",
        "KickOff":    "kickoff",
        "Penalty":    "penalty_kick",
    }
    pred_sp = result[result["set_piece_type"].notna()].copy()
    pred_sp["label"] = pred_sp["set_piece_type"].map(label_map)
    pred_sp = pred_sp.dropna(subset=["label"])
    return pred_sp[["period_id", "timestamp", "trigger_player", "label"]].reset_index(drop=True)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def confusion_matrix_sp(
    gt_sp: pd.DataFrame,
    pred_sp: pd.DataFrame,
    window: float = SP_WINDOW,
) -> pd.DataFrame:
    """
    Build a confusion matrix for set piece detection.

    Greedy 1:1 time-window matching between GT and pred (same as evaluate()).
    Matched pairs are tallied as (GT label → Pred label).
    Unmatched GT rows become (GT label → 'missed').
    Unmatched pred rows become ('extra' → Pred label).

    Returns
    -------
    cm : DataFrame indexed by GT label (rows) × pred label (cols)
         rows: SP_LABELS + ['extra']
         cols: SP_LABELS + ['missed']
    """
    row_labels = SP_LABELS + ["extra"]
    col_labels = SP_LABELS + ["missed"]
    cm = pd.DataFrame(0, index=row_labels, columns=col_labels)

    gt_work   = gt_sp.copy().reset_index(drop=True)
    pred_work = pred_sp.copy().reset_index(drop=True)
    matched_pred: set[int] = set()
    matched_gt:   set[int] = set()

    for gi, gr in gt_work.iterrows():
        cand = pred_work[
            (pred_work["period_id"] == gr["period_id"]) &
            (abs(pred_work["timestamp"] - gr["timestamp"]) <= window)
        ]
        cand = cand[~cand.index.isin(matched_pred)]
        if cand.empty:
            continue
        best_pi = (cand["timestamp"] - gr["timestamp"]).abs().idxmin()
        matched_gt.add(gi)
        matched_pred.add(best_pi)
        gt_lbl   = gr["label"]
        pred_lbl = pred_work.at[best_pi, "label"]
        if gt_lbl in cm.index and pred_lbl in cm.columns:
            cm.at[gt_lbl, pred_lbl] += 1

    # Unmatched GT → missed
    for gi, gr in gt_work.iterrows():
        if gi not in matched_gt:
            gt_lbl = gr["label"]
            if gt_lbl in cm.index:
                cm.at[gt_lbl, "missed"] += 1

    # Unmatched pred → extra
    for pi, pr in pred_work.iterrows():
        if pi not in matched_pred:
            pred_lbl = pr["label"]
            if pred_lbl in cm.columns:
                cm.at["extra", pred_lbl] += 1

    return cm


def confusion_matrix_open(
    gt_open: pd.DataFrame,
    pred_open: pd.DataFrame,
    window: float = PAPER_WINDOW,
    use_player: bool = True,
) -> pd.DataFrame:
    """
    Build a confusion matrix for open play event detection.

    Greedy 1:1 matching using the same criteria as evaluate_paper():
    period_id match + time window (+ optional player match).
    Matched pairs: GT label → Pred label.
    Unmatched GT → (GT label, 'missed').
    Unmatched pred → ('extra', Pred label).

    Returns
    -------
    cm : DataFrame indexed by GT label (rows) × pred label (cols)
         rows: OPEN_LABELS + ['extra']
         cols: OPEN_LABELS + ['missed']
    """
    row_labels = OPEN_LABELS + ["extra"]
    col_labels = OPEN_LABELS + ["missed"]
    cm = pd.DataFrame(0, index=row_labels, columns=col_labels)

    gt_work   = gt_open.copy().reset_index(drop=True)
    pred_work = pred_open.copy().reset_index(drop=True)
    matched_pred: set[int] = set()
    matched_gt:   set[int] = set()

    for gi, gr in gt_work.iterrows():
        cand = pred_work[
            (pred_work["period_id"] == gr["period_id"]) &
            (abs(pred_work["timestamp"] - gr["timestamp"]) <= window)
        ]
        if use_player and "object_id" in gr and "event_player" in pred_work.columns:
            cand = cand[cand["event_player"] == gr["object_id"]]
        cand = cand[~cand.index.isin(matched_pred)]
        if cand.empty:
            continue
        best_pi = (cand["timestamp"] - gr["timestamp"]).abs().idxmin()
        matched_gt.add(gi)
        matched_pred.add(best_pi)
        gt_lbl   = gr["label"]
        pred_lbl = pred_work.at[best_pi, "label"]
        if gt_lbl in cm.index and pred_lbl in cm.columns:
            cm.at[gt_lbl, pred_lbl] += 1

    for gi, gr in gt_work.iterrows():
        if gi not in matched_gt:
            gt_lbl = gr["label"]
            if gt_lbl in cm.index:
                cm.at[gt_lbl, "missed"] += 1

    for pi, pr in pred_work.iterrows():
        if pi not in matched_pred:
            pred_lbl = pr["label"]
            if pred_lbl in cm.columns:
                cm.at["extra", pred_lbl] += 1

    return cm


# ── Single-match set-piece evaluation ────────────────────────────────────────

def eval_sp_match(
    match: SportecData,
    result: pd.DataFrame,
    sp_window: float = SP_WINDOW,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Run set-piece evaluation for a single match.

    Returns
    -------
    sp_rows : list of per-label result dicts
    cm      : confusion matrix DataFrame
    """
    gt_sp   = prepare_gt_sp(match)
    pred_sp = prepare_pred_sp(result)

    sp_rows = [evaluate(gt_sp, pred_sp, lbl, window=sp_window) for lbl in SP_LABELS]
    cm      = confusion_matrix_sp(gt_sp, pred_sp, window=sp_window)

    return sp_rows, cm


# ── All-matches set-piece evaluation ─────────────────────────────────────────

def eval_sp_all_matches(
    match_ids: list[str],
    results: dict[str, pd.DataFrame],
    sp_window: float = SP_WINDOW,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate set pieces across all matches.

    Parameters
    ----------
    match_ids : list of match ID strings
    results   : dict mapping match_id → SetPieceDetector result DataFrame
    sp_window : time tolerance in seconds (default 10)

    Returns
    -------
    summary : aggregated P/R/F1 per label  (from aggregate())
    cm      : summed confusion matrix across all matches
    """
    all_sp: list[dict] = []
    row_labels = SP_LABELS + ["extra"]
    col_labels = SP_LABELS + ["missed"]
    total_cm = pd.DataFrame(0, index=row_labels, columns=col_labels)

    for mid in match_ids:
        if verbose:
            print(f"Evaluating {mid}...", end=" ", flush=True)
        m = SportecData(mid, load_tracking=False)
        sp_rows, cm = eval_sp_match(m, results[mid], sp_window=sp_window)
        for row in sp_rows:
            row["match"] = mid
            all_sp.append(row)
        total_cm = total_cm.add(cm, fill_value=0).astype(int)
        if verbose:
            print("done")

    summary = aggregate(all_sp)
    return summary, total_cm


# ── RDP parquet 기반 오픈플레이 평가 ─────────────────────────────────────────

def prepare_pred_open(open_result: pd.DataFrame) -> pd.DataFrame:
    """
    OpenPlayEventDetector 출력에서 오픈플레이 pred DataFrame을 생성.

    open_result must have: period_id, timestamp, event_name, event_player

    Returns
    -------
    pred : columns: period_id, timestamp, event_player, label
    """
    pred_map = {
        "Pass":            "pass",
        "Cross":           "cross",
        "ShotOnTarget":    "shot",
        "ShotOffTarget":   "shot",
    }
    pred = open_result[open_result["event_name"].isin(pred_map)].copy()
    pred["label"] = pred["event_name"].map(pred_map)
    return pred[["period_id", "timestamp", "event_player", "label"]].reset_index(drop=True)


def eval_match_rdp(
    match_id: str,
    open_result: pd.DataFrame,
    open_window: float = PAPER_WINDOW,
    use_player: bool = True,
    cache_dir: str = "cache",
) -> list[dict]:
    """
    단일 경기 오픈플레이 평가 (event_rdp parquet GT 사용).

    Parameters
    ----------
    match_id    : 경기 ID (예: 'J03WN1')
    open_result : OpenPlayEventDetector 출력 DataFrame
    open_window : 매칭 허용 시간 (초)
    use_player  : 선수 일치 조건 사용 여부
    cache_dir   : 사용하지 않음 (호환용)

    Returns
    -------
    open_rows : list of per-label result dicts
    """
    from tools.parquet_loader import load_gt_open

    gt_open = load_gt_open(match_id)
    pred_open = prepare_pred_open(open_result)

    open_rows = [
        evaluate_paper(gt_open, pred_open, lbl, window=open_window, use_player=use_player)
        for lbl in OPEN_LABELS
    ]
    return open_rows


def eval_all_matches_rdp(
    match_ids: list[str],
    open_results: dict[str, pd.DataFrame],
    open_window: float = PAPER_WINDOW,
    use_player: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    전체 경기 오픈플레이 평가 (event_rdp parquet GT 사용).

    Parameters
    ----------
    match_ids    : 경기 ID 리스트
    open_results : {match_id → OpenPlayEventDetector 출력 DataFrame}
    open_window  : 매칭 허용 시간 (초)
    use_player   : 선수 일치 조건 사용 여부
    verbose      : 경기별 진행 출력 여부

    Returns
    -------
    summary : aggregated P/R/F1 per label
    """
    all_open: list[dict] = []

    for mid in match_ids:
        if verbose:
            print(f"Evaluating {mid}...", end=" ", flush=True)
        rows = eval_match_rdp(mid, open_results[mid], open_window=open_window, use_player=use_player)
        for row in rows:
            row["match"] = mid
            all_open.append(row)
        if verbose:
            print("done")

    return aggregate(all_open)
