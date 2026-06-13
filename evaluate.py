"""Evaluate kick predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.AutoEvent.evaluate import SP_WINDOW, OPEN_LABELS, evaluate_paper, prepare_pred_open
from eval.run_dedup_eval import dedup_predictions_v2
from eval.run_raw_eval import gt_extend
from scipy.optimize import linear_sum_assignment
from config import KICK_LABELS, SET_PIECE_LABELS

def extend_kick_ground_truth(raw_gt: pd.DataFrame) -> pd.DataFrame:
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

def extend_set_piece_ground_truth(raw_gt: pd.DataFrame) -> pd.DataFrame:
    """Build raw-GT set piece labels aligned to AutoEvent label space."""
    label_to_true_map = {
        "ThrowIn": "throw_in",
        "GoalKick": "goal_kick",
        "CornerKick": "corner_kick",
        "FreeKick": "free_kick",
        "KickOff": "kickoff",
        "Penalty": "penalty_kick",
    }

    rows = []
    for _, r in raw_gt.iterrows():
        label = label_to_true_map.get(r["event_kind"])
        if label is None:
            continue

        actor = r.get("player_id")
        if not isinstance(actor, str):
            actor = r.get("play_player")
        if not isinstance(actor, str):
            actor = "unknown"

        ev_ts = r.get("sync_ts", r["timestamp"])
        if pd.isna(ev_ts):
            ev_ts = r["timestamp"]

        rows.append({
            "period_id": int(r["period_id"]),
            "timestamp": float(ev_ts),
            "object_id": actor,
            "label": label,
        })
        
    return pd.DataFrame(rows)

def drop_duplicated_kick(df: pd.DataFrame, duplicated_window: float) -> pd.DataFrame:
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
                if t_j - t_i > duplicated_window:
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

def evaluate(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label: str,
    window: float = 1.0,
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
    true_df   = gt_df[gt_df["label"] == label][["period_id", "timestamp", "object_id"]].copy().reset_index(drop=True)
    pred_df   = pred_df[pred_df["label"] == label][["period_id", "timestamp", "event_player"]].copy().reset_index(drop=True)

    matched_gt   = set()
    matched_pred = set()

    if use_hungarian:
        # Per-period optimal assignment
        for period in true_df["period_id"].unique():
            gi_idx = true_df.index[true_df["period_id"] == period].tolist()
            pi_idx = pred_df.index[pred_df["period_id"] == period].tolist()
            if not gi_idx or not pi_idx:
                continue
            # Cost = |Δt|, infeasible (>window or player mismatch) → big cost
            INF = 1e6
            cost = np.full((len(gi_idx), len(pi_idx)), INF)
            for ii, gi in enumerate(gi_idx):
                gt = true_df.at[gi, "timestamp"]
                gp = true_df.at[gi, "object_id"]
                for jj, pi in enumerate(pi_idx):
                    pt = pred_df.at[pi, "timestamp"]
                    if abs(pt - gt) > window:
                        continue
                    if use_player and pred_df.at[pi, "event_player"] != gp:
                        continue
                    cost[ii, jj] = abs(pt - gt)
            row_idx, col_idx = linear_sum_assignment(cost)

            for ii, jj in zip(row_idx, col_idx):
                if cost[ii, jj] < INF:
                    matched_gt.add(gi_idx[ii])
                    matched_pred.add(pi_idx[jj])
    else:
        for gi, gr in true_df.iterrows():
            cand = pred_df[
                (pred_df["period_id"] == gr["period_id"]) &
                (abs(pred_df["timestamp"] - gr["timestamp"]) <= window)
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
    fp = len(pred_df) - len(matched_pred)
    fn = len(true_df) - tp
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "label": label, "GT": len(true_df), "Pred": len(pred_df),
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3),
    }

def evaluate_set_piece(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label: str,
    window: float = SP_WINDOW,
    use_player: bool = False,
) -> dict:
    """Set piece evaluation with optional player constraint."""
    true_df = gt_df[gt_df["label"] == label][["period_id", "timestamp", "object_id"]].copy().reset_index(drop=True)
    pred_df = pred_df[pred_df["label"] == label][["period_id", "timestamp", "trigger_player"]].copy().reset_index(drop=True)

    matched_gt = set()
    matched_pred = set()

    for gi, gr in true_df.iterrows():
        cand = pred_df[
            (pred_df["period_id"] == gr["period_id"]) &
            (abs(pred_df["timestamp"] - gr["timestamp"]) <= window)
        ]
        if use_player:
            cand = cand[cand["trigger_player"] == gr["object_id"]]
        cand = cand[~cand.index.isin(matched_pred)]
        if cand.empty:
            continue
        best_pi = (cand["timestamp"] - gr["timestamp"]).abs().idxmin()
        matched_gt.add(gi)
        matched_pred.add(best_pi)

    tp = len(matched_gt)
    fp = len(pred_df) - len(matched_pred)
    fn = len(true_df) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return {
        "label": label, "GT": len(true_df), "Pred": len(pred_df),
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3),
    }
    
def get_metrics(results: pd.DataFrame, type: str, matching: str = "Greedy") -> pd.DataFrame:
    metrics = results.groupby("label")[["GT", "Pred", "TP", "FP", "FN"]].sum()
    metrics["Precision"] = (metrics["TP"] / (metrics["TP"] + metrics["FP"])).round(3)
    metrics["Recall"]    = (metrics["TP"] / (metrics["TP"] + metrics["FN"])).round(3)
    d = (metrics["Precision"] + metrics["Recall"]).replace(0, 1)
    metrics["F1"] = (2 * metrics["Precision"] * metrics["Recall"] / d).round(3)
    tp, fp, fn = int(metrics["TP"].sum()), int(metrics["FP"].sum()), int(metrics["FN"].sum())
    precision  = tp / (tp + fp) if (tp + fp) else 0.0
    recall  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n[{type} | {matching}]")
    for label in metrics.index:
        if label in metrics.index:
            row = metrics.loc[label]
            print(f"{label.capitalize():<6}, P={row['Precision']:.3f}, R={row['Recall']:.3f}, F1={row['F1']:.3f}")
    macro_f1 = metrics["F1"].mean()
    print(f"  Macro: F1={macro_f1:.3f}")
    print(f"  Micro: F1={f1:.3f}", flush=True)
    
def run_kick(df: pd.DataFrame, raw_data_path: Path, evaluated_window: float = 1.0, duplicated_window: float = 2.0, use_player: bool = False) -> None:
    """Kick evaluation — same pipeline as run_compare.py quick_eval.

    Bypasses _load_person_map (not needed when use_player=False).
    """

    if duplicated_window > 0:
        df = drop_duplicated_kick(df, duplicated_window)

    df = df.copy()
    df["event_name"]   = df["pred_label"].map({"none": None, "pass": "Pass", "cross": "Cross", "shot": "ShotOffTarget"})
    df["event_player"] = df["loss_player"]
    df["event_team"]   = df["loss_team"]

    label_to_pred_map = {"Pass": "pass", "Cross": "cross", "ShotOnTarget": "shot", "ShotOffTarget": "shot"}
    rows = []
    for match_id in tqdm(sorted(df["match_id"].unique()), desc="Evaluating Kick"):
        pred_kick_df = df[(df["match_id"] == match_id) & (df["event_name"].isin(label_to_pred_map.keys()))].reset_index(drop=True)
        pred_kick_df["label"] = pred_kick_df["event_name"].map(label_to_pred_map)
        
        true_df = pd.read_parquet(Path(raw_data_path) / f"{match_id}.parquet")
        true_kick_df = extend_kick_ground_truth(true_df)
        true_kick_df = true_kick_df[true_kick_df["label"] != "none"].reset_index(drop=True)

        elastic_df = pd.read_parquet(Path(raw_data_path).parent / "elastic" / f"{match_id}" / f"event.parquet")
        elastic_df["label"] = elastic_df["event_type"]
        elastic_df["object_id"] = elastic_df["player_id"]
        elastic_df["timestamp"] = elastic_df["timestamp"].apply(lambda x: sum(float(t) * 60 ** i for i, t in enumerate(reversed(x.split(":")))))

        for label in KICK_LABELS:
            if label == "none":
                continue

            #row = evaluate(true_kick_df, pred_kick_df, label, window=evaluated_window, use_player=False)
            row = evaluate(elastic_df, pred_kick_df, label, window=evaluated_window, use_player=True)
            row["match"] = match_id
            rows.append(pd.Series(row))

    get_metrics(pd.DataFrame(rows), type="Kick")

def run_ablation_kick(df: pd.DataFrame, raw_data_path: Path, win: float = 2.0, model_name: str = "XGB") -> None:
    """Stage-3 ablation kick evaluation (Table 3): paper-style Hungarian matching, +/-1s window.

    Unlike run_kick (which matches against the elastic event log), this matches
    predictions against gt_extend(raw_gt) via evaluate_paper — the same protocol
    used to evaluate the TabPFN/CatBoost/TabNet/FT-Transformer/TabTransformer ablations.
    """
    if win > 0:
        df = dedup_predictions_v2(df, win)

    df = df.copy()
    df["event_name"]   = df["pred_label"].map(
        {"none": pd.NA, "pass": "Pass", "cross": "Cross", "shot": "ShotOffTarget"}
    )
    df["event_player"] = df["loss_player"]
    df["event_team"]   = df["loss_team"]

    rows = []
    for match_id in sorted(df["match_id"].unique()):
        s         = df[df["match_id"] == match_id]
        pred_open = prepare_pred_open(s)
        raw = pd.read_parquet(Path(raw_data_path) / f"{match_id}.parquet")
        if "sync_ts" in raw.columns:
            raw = raw.copy()
            raw["timestamp"] = raw["sync_ts"].fillna(raw["timestamp"])
        gt = gt_extend(raw)
        for label in OPEN_LABELS:
            row = evaluate_paper(gt, pred_open, label, window=1.0, use_player=True)
            row["match"] = match_id
            rows.append(pd.Series(row))

    get_metrics(pd.DataFrame(rows), type=f"Kick | {model_name} | w={win}", matching="Hungarian")

def run_set_piece(df: pd.DataFrame, raw_data_path: Path, evaluated_window: float = 1.0, duplicated_window: float = 2.0, use_player: bool = False) -> None:
    """Set-piece evaluation — same matching protocol as run_dedup_eval.run_eval.

    GT: true labels stored in the test parquet (assigned by elastic GT labeler).
    Matching: evaluate_paper, window=±10s, use_player=False.
    """

    label_to_pred_map = {"ThrowIn": "throw_in", "GoalKick": "goal_kick", "CornerKick": "corner_kick", "FreeKick": "free_kick", "FreeKick?": "free_kick", "KickOff": "kickoff", "Penalty": "penalty_kick"}
    rows = []
    for match_id in tqdm(sorted(df["match_id"].unique()), desc="Evaluating Set Pieces"):
        pred_set_piece_df = df[(df["match_id"] == match_id) & (df["set_piece_type"].notna())].reset_index(drop=True)
        pred_set_piece_df["label"] = pred_set_piece_df["set_piece_type"].map(label_to_pred_map)
        pred_set_piece_df = pred_set_piece_df.dropna(subset=["label"])
        
        true_set_piece_df = pd.read_parquet(Path(raw_data_path) / f"{match_id}.parquet")
        true_set_piece_df = extend_set_piece_ground_truth(true_set_piece_df)
        
        for label in SET_PIECE_LABELS:
            row = evaluate_set_piece(true_set_piece_df, pred_set_piece_df, label, window=evaluated_window, use_player=use_player)
            row["match"] = match_id
            rows.append(pd.Series(row))
            
    get_metrics(pd.DataFrame(rows), type="Set Piece")

def run_custom_rule(df: pd.DataFrame, raw_data_path: Path, evaluated_window: float = 1.0, duplicated_window: float = 2.0, use_player: bool = False) -> None:
    """Custom rule evaluation — same matching protocol as run_dedup_eval.run_eval.

    GT: true labels stored in the test parquet (assigned by elastic GT labeler).
    Matching: evaluate_paper, window=±10s, use_player=False.
    """
    rows = []
    for match_id in tqdm(sorted(df["match_id"].unique()), desc="Evaluating Custom Rule"):
        pred_kick_df = df[df["match_id"] == match_id].reset_index(drop=True)
        pred_kick_df["label"] = pred_kick_df["subtype"]
        pred_kick_df["event_player"] = pred_kick_df["player"]

        true_df = pd.read_parquet(Path(raw_data_path) / f"{match_id}.parquet")
        true_kick_df = extend_kick_ground_truth(true_df)
        true_kick_df = true_kick_df[true_kick_df["label"] != "none"].reset_index(drop=True)

        elastic_df = pd.read_parquet(Path(raw_data_path).parent / "elastic" / f"{match_id}" / f"event.parquet")
        elastic_df["label"] = elastic_df["event_type"]
        elastic_df["object_id"] = elastic_df["player_id"]

        # timestamp (00:00:00) -> seconds
        elastic_df["timestamp"] = elastic_df["timestamp"].apply(lambda x: sum(float(t) * 60 ** i for i, t in enumerate(reversed(x.split(":")))))
        # print(pred_kick_df)
        # print(true_kick_df)
        # print(elastic_df)
        # print(pred_kick_df["subtype"].value_counts())
        # print(true_kick_df["label"].value_counts())
        # print(elastic_df["event_type"].value_counts())

        for label in KICK_LABELS:
            if label == "none":
                continue

            row = evaluate(true_kick_df, pred_kick_df, label, window=evaluated_window, use_player=use_player)
            #row = evaluate(elastic_df, pred_kick_df, label, window=evaluated_window, use_player=use_player)
            row["match"] = match_id
            rows.append(pd.Series(row))

    get_metrics(pd.DataFrame(rows), type="Custom Rule")

def main() -> None:
    """
        python evaluate.py \
        --raw_data_path ./data/dfl/processed/raw_gt_synced \
        --kick_pred_path ./data/dfl/ml/predictions/xgb_predictions.parquet \
        --set_piece_pred_path ./data/dfl/ml/detection

        python evaluate.py \
        --raw_data_path ./data/dfl/processed/raw_gt_synced \
        --custom_rule_pred_path ./data/dfl/custom_rule/predictions/custom_rule_predictions.parquet

        # Table 3 ablation (Stage-3 classifiers, Hungarian +/-1s matching):
        python evaluate.py \
        --raw_data_path ./data/dfl/processed/raw_gt_synced \
        --xgb_pred ./data/dfl/ml/predictions/xgb_predictions.parquet \
        --tabpfn_pred ./data/dfl/ml/predictions/tabpfn_predictions.parquet \
        --catboost_pred ./data/dfl/ml/predictions/catboost_predictions.parquet \
        --tabnet_pred ./data/dfl/ml/predictions/tabnet_predictions.parquet \
        --ft_transformer_pred ./data/dfl/ml/predictions/fttransformer_predictions.parquet \
        --tab_transformer_pred ./data/dfl/ml/predictions/tabtransformer_predictions.parquet \
        --rule_pred ./data/dfl/ml/predictions/rule_predictions.parquet
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="./data/dfl/processed/raw_gt_synced")
    parser.add_argument("--kick_pred_path", default=None)
    parser.add_argument("--set_piece_pred_path", default=None)
    parser.add_argument("--custom_rule_pred_path", default=None)
    parser.add_argument("--duplicated_window", type=float, default=2.0, help="Duplicate window in seconds")
    parser.add_argument("--evaluated_window", type=float, default=1.0, help="Evaluation matching window in seconds")
    parser.add_argument("--use_player", action="store_true", help="Whether to require player match in evaluation")

    # Stage-3 ablation (Table 3): per-frame prediction parquets, evaluated via paper-style Hungarian matching
    parser.add_argument("--xgb_pred", default=None)
    parser.add_argument("--tabpfn_pred", default=None)
    parser.add_argument("--catboost_pred", default=None)
    parser.add_argument("--tabnet_pred", default=None)
    parser.add_argument("--ft_transformer_pred", default=None)
    parser.add_argument("--tab_transformer_pred", default=None)
    parser.add_argument("--rule_pred", default=None)
    parser.add_argument("--ablation_win", type=float, default=2.0, help="Dedup window (s) for ablation kick eval")
    args = parser.parse_args()

    ablation_preds = {
        "XGB": args.xgb_pred,
        "TabPFN": args.tabpfn_pred,
        "CatBoost": args.catboost_pred,
        "TabNet": args.tabnet_pred,
        "FT-Transformer": args.ft_transformer_pred,
        "TabTransformer": args.tab_transformer_pred,
        "Rule": args.rule_pred,
    }

    if not any([args.kick_pred_path, args.set_piece_pred_path, args.custom_rule_pred_path, *ablation_preds.values()]):
        raise ValueError("At least one of --kick_pred_path, --set_piece_pred_path, --custom_rule_pred_path, or an ablation --*_pred argument must be provided.")

    if args.kick_pred_path:
        kick_pred_df = pd.read_parquet(args.kick_pred_path)
        print(f"Kick predictions: {len(kick_pred_df)} rows, matche_ids: {sorted(kick_pred_df['match_id'].unique())}")
        run_kick(kick_pred_df, raw_data_path=args.raw_data_path,
                 evaluated_window=args.evaluated_window, duplicated_window=args.duplicated_window, use_player=args.use_player)

    if args.custom_rule_pred_path:
        custom_rule_pred_df = pd.read_parquet(args.custom_rule_pred_path)
        print(f"\nCustom Rule predictions: {len(custom_rule_pred_df)} rows, matche_ids: {sorted(custom_rule_pred_df['match_id'].unique())}")
        run_custom_rule(custom_rule_pred_df, raw_data_path=args.raw_data_path, 
                        evaluated_window=args.evaluated_window, duplicated_window=args.duplicated_window, use_player=args.use_player)

    if args.set_piece_pred_path:
        set_piece_pred_df = []
        for set_piece_path in tqdm(list(Path(args.set_piece_pred_path).glob("*")), desc="Loading Set Piece Predictions"):
            match_df = pd.read_parquet(Path(set_piece_path) / "set_piece.parquet")
            match_df["match_id"] = set_piece_path.stem
            set_piece_pred_df.append(match_df)

        set_piece_pred_df = pd.concat(set_piece_pred_df, ignore_index=True)
        print(f"\nSet-piece predictions: {len(set_piece_pred_df)} rows, matche_ids: {sorted(set_piece_pred_df['match_id'].unique())}")
        run_set_piece(set_piece_pred_df, raw_data_path=args.raw_data_path,
                      evaluated_window=10, duplicated_window=args.duplicated_window, use_player=True)

    for model_name, pred_path in ablation_preds.items():
        if not pred_path:
            continue
        df = pd.read_parquet(pred_path)
        print(f"\n{model_name} predictions: {len(df)} rows  matches={sorted(df['match_id'].unique())}")
        run_ablation_kick(df, raw_data_path=args.raw_data_path, win=args.ablation_win, model_name=model_name)

if __name__ == "__main__":
    main()
