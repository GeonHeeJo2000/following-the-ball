from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from collections import defaultdict

from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from models.AutoEvent.possession import PossessionDetector
from models.AutoEvent.setpiece import SetPieceDetector
from models.AutoEvent.openplay import OpenPlayDetector

from config import (
    PITCH_X,
    PITCH_Y,
    PA_X,
    PA_Y_MIN,
    PA_Y_MAX,
    GOAL_HALF_W,
    TRACKING_FPS
)

class BaseEventDataset(ABC):
    task_name: str

    def __init__(self, data_path: Path | None = None, cache_path: Path | None = None, save_path: Path | None = None) -> None:
        self.data_path = data_path
        self.raw_data_path = os.path.join(data_path, "raw_gt_synced")
        self.elastic_path = os.path.join(data_path, "elastic")
        self.cache_path = cache_path
        self.save_path = save_path
        
        Path(cache_path, self.task_name).mkdir(parents=True, exist_ok=True)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        self.match_ids = sorted([path.split(".")[0] for path in os.listdir(self.elastic_path)])
        self.match_ids = [
            id for id in self.match_ids
            if os.path.exists(os.path.join(self.elastic_path, id, "tracking.parquet"))
        ]
        self.train_match_ids, self.valid_match_ids, self.test_match_ids = self.split_match_ids()
            
    @abstractmethod
    def compute_features_and_labels(self, match_id: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def assign_labels(self, feats: list[dict[str, Any]], raw_gt: pd.DataFrame, goalkeepers: list[str], window_sec: float) -> tuple[list[int], list[int]]:
        raise NotImplementedError

    def split_match_ids(
        self,
        # train_ratio: float = 5 / 7, # (6 matches train, 1 match valid, 1 match test)
        # valid_ratio: float = 1 / 7,
        # test_ratio: float = 1 / 7,
        train_ratio: float = 0.8, # (6 matches train, 1 match valid, 1 match test)
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[str], list[str], list[str]]:

        train_end, valid_end = int(len(self.match_ids) * train_ratio), int(len(self.match_ids) * (train_ratio + valid_ratio))
        train_match_ids, valid_match_ids, test_match_ids = self.match_ids[:train_end], self.match_ids[train_end:valid_end], self.match_ids[valid_end:]
        print(f"Match IDs: {self.match_ids}")
        print(f"Total: {len(self.match_ids)} samples, Train: {len(train_match_ids)} samples, Valid: {len(valid_match_ids)} samples, Test: {len(test_match_ids)} samples")
        
        # Save info.txt
        info_path = os.path.join(self.cache_path, self.task_name, "info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"task: {self.task_name}\n")
            
            f.write(f"total_matches: {len(self.match_ids)}\n")
            f.write(f"train_matches: {len(train_match_ids)}\n")
            f.write(f"valid_matches: {len(valid_match_ids)}\n")
            f.write(f"test_matches: {len(test_match_ids)}\n")
            
            f.write(f"train_match_ids: {train_match_ids}\n")
            f.write(f"valid_match_ids: {valid_match_ids}\n")
            f.write(f"test_match_ids: {test_match_ids}\n")
            
            f.write(f"train_ratio: {train_ratio}\n")
            f.write(f"valid_ratio: {valid_ratio}\n")
            f.write(f"test_ratio: {test_ratio}\n")
            
            f.write(f"seed: {seed}\n")
        print(f"Saved dataset split info to {info_path}")
        
        return train_match_ids, valid_match_ids, test_match_ids

    def detect_frames(self, match_id: str) -> pd.DataFrame:
        tracking_path = os.path.join(self.elastic_path, match_id, "tracking.parquet")
        
        base_detections_path = os.path.join(self.cache_path, "detection", match_id)
        os.makedirs(base_detections_path, exist_ok=True)
        possession_path = os.path.join(base_detections_path, "possession.parquet")
        set_piece_path = os.path.join(base_detections_path, "set_piece.parquet")
        open_play_path = os.path.join(base_detections_path, "open_play.parquet")
        
        tracking = pd.read_parquet(tracking_path)
        
        if not os.path.exists(possession_path):
            possession_detector = PossessionDetector(tracking, pre_smoothed=True)
            possession = possession_detector.run()
            possession.to_parquet(possession_path, index=False)
        else:
            possession = pd.read_parquet(possession_path)

        if not os.path.exists(set_piece_path):
            set_piece_detector = SetPieceDetector(possession)
            set_piece = set_piece_detector.run()
            set_piece.to_parquet(set_piece_path, index=False)
        else:
            set_piece = pd.read_parquet(set_piece_path)
        
        if not os.path.exists(open_play_path):
            open_play_detector = OpenPlayDetector(set_piece)
            open_play = open_play_detector.run()
            open_play.to_parquet(open_play_path, index=False)
        else:
            open_play = pd.read_parquet(open_play_path)

        return possession, set_piece, open_play
    
    def prepare_datasets(
        self,
        match_ids: list[str],
        split_name: str,
    ) -> pd.DataFrame:
        
        def process_match(match_id: str, cache_path: str, task_name: str) -> pd.DataFrame:
            split_path = os.path.join(cache_path, task_name, f"{match_id}.parquet")
            if os.path.exists(split_path):
                print(f"Loading cached match {match_id} for split {split_name} from {split_path}")
                return pd.read_parquet(split_path)
            else:
                print(f"Building match {match_id} for split {split_name}")
                frame = self.compute_features_and_labels(match_id)
                frame.to_parquet(split_path, index=False)
                return frame

        n_jobs = min(os.cpu_count(), len(match_ids))
        if n_jobs == 1:
            frames = [process_match(mid, self.cache_path, self.task_name) for mid in tqdm(match_ids, desc=f"Single Processing matches for {split_name}")]
        else:
            print(f"Multi Processing {len(match_ids)} matches for split {split_name} using {n_jobs} parallel jobs...\n")
            with tqdm_joblib(tqdm_joblib(total=len(match_ids), desc=f"Preparing {split_name} dataset")):
                frames = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
                    delayed(process_match)(mid, self.cache_path, self.task_name) 
                    for mid in match_ids
                )

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def extract_features(poss: pd.DataFrame, idx: int, players: list[str], include_destination: bool = False) -> dict[str, Any]:
        row = poss.iloc[idx]
        loss_team = row["loss_team"]
        loss_player = row["loss_player"]

        bx = float(row.get("ball_x"))
        by = float(row.get("ball_y"))
        bz = float(row.get("ball_z"))
        bvx = float(row.get("ball_vx"))
        bvy = float(row.get("ball_vy"))
        bspeed = float(row.get("ball_speed"))
        baccel = float(row.get("ball_accel"))

        gx, gy = (PITCH_X, PITCH_Y / 2) if loss_team == "home" else (0.0, PITCH_Y / 2)
        dist_to_goal = float(np.hypot(bx - gx, by - gy))
        goal_vec_x = gx - bx
        goal_vec_y = gy - by
        g_norm = float(np.hypot(goal_vec_x, goal_vec_y) + 1e-9)
        v_norm = float(np.hypot(bvx, bvy) + 1e-9)
        cos_to_goal = float((bvx * goal_vec_x + bvy * goal_vec_y) / (v_norm * g_norm))
        sideline_dist = float(min(abs(by), abs(PITCH_Y - by)))

        in_attacking_half = (
            (loss_team == "home" and bx > PITCH_X / 2)
            or (loss_team == "away" and bx < PITCH_X / 2)
        )
        rel_x = bx if loss_team == "home" else PITCH_X - bx
        in_shot_zone = rel_x >= 75 and PA_Y_MIN <= by <= PA_Y_MAX
        in_cross_zone = rel_x >= 75 and (by < PA_Y_MIN or by > PA_Y_MAX)
        in_attacking_pa = rel_x >= (PITCH_X - PA_X) and PA_Y_MIN <= by <= PA_Y_MAX

        if isinstance(loss_player, str):
            lp_x = float(row.get(f"{loss_player}_x"))
            lp_y = float(row.get(f"{loss_player}_y"))
            lp_speed = float(row.get(f"{loss_player}_speed"))
        else:
            lp_x = lp_y = np.nan
            lp_speed = 0.0

        opp_team = "away" if loss_team == "home" else "home"
        nearest_opp_dist = float("inf")
        n_opps_in_5m = 0
        n_attackers_in_pa = 0

        speed_sq = bvx * bvx + bvy * bvy
        has_direction = speed_sq > 0.25
        if has_direction:
            ball_speed = float(np.sqrt(speed_sq))
            nx, ny = bvx / ball_speed, bvy / ball_speed
        else:
            nx = ny = 0.0

        team_in_cone15 = 0
        team_in_cone30 = 0
        opp_in_cone30 = 0
        nearest_team_along = 50.0
        nearest_opp_along = 50.0
        nearest_team_player = None

        for player in players:
            ptype = "home" if player.startswith("home_") else "away"
            px = float(row.get(f"{player}_x"))
            py = float(row.get(f"{player}_y"))
            if np.isnan(px) or np.isnan(py):
                continue

            d = float(np.hypot(px - bx, py - by))
            if ptype == opp_team:
                nearest_opp_dist = min(nearest_opp_dist, d)
                if d < 5.0:
                    n_opps_in_5m += 1
            else:
                attacker_rel_x = px if loss_team == "home" else PITCH_X - px
                if attacker_rel_x >= (PITCH_X - PA_X) and PA_Y_MIN <= py <= PA_Y_MAX:
                    n_attackers_in_pa += 1

            if has_direction and d > 0.5:
                dx_ = px - bx
                dy_ = py - by
                proj = nx * dx_ + ny * dy_
                if proj > 0:
                    perp = abs(-ny * dx_ + nx * dy_)
                    angle = float(np.degrees(np.arctan2(perp, proj)))
                    in_cone15 = angle <= 15.0 and proj < 50.0
                    in_cone30 = angle <= 30.0 and proj < 50.0
                    is_self = player == loss_player
                    if not is_self and ptype != opp_team:
                        if in_cone15:
                            team_in_cone15 += 1
                        if in_cone30:
                            team_in_cone30 += 1
                            if proj < nearest_team_along:
                                nearest_team_along = proj
                                nearest_team_player = player
                    elif ptype == opp_team:
                        if in_cone30:
                            opp_in_cone30 += 1
                            if proj < nearest_opp_along:
                                nearest_opp_along = proj

        if nearest_opp_dist == float("inf"):
            nearest_opp_dist = 30.0

        future = poss.iloc[idx:min(idx + 25, len(poss))]
        ball_z_max_1s = float(future["ball_z"].max() or 0)

        start_p = max(0, idx - 25)
        past = poss.iloc[start_p:idx]
        if len(past) > 0:
            past_speed_mean = float(past["ball_speed"].mean() or 0)
            past_speed_max = float(past["ball_speed"].max() or 0)
            past_accel_max = float(past["ball_accel"].abs().max() or 0)
            past_z_max = float(past["ball_z"].max() or 0)
        else:
            past_speed_mean = past_speed_max = past_accel_max = past_z_max = 0.0

        future_2s = poss.iloc[idx:min(idx + 50, len(poss))]
        if len(future_2s) > 0:
            future_speed_max = float(future_2s["ball_speed"].max() or 0)
            future_z_max_2s = float(future_2s["ball_z"].max() or 0)
            future_air_frames = int((future_2s["ball_z"] > 0.3).sum())
        else:
            future_speed_max = future_z_max_2s = 0.0
            future_air_frames = 0

        if len(future) >= 5:
            fvx = future["ball_vx"].fillna(0).to_numpy()
            fvy = future["ball_vy"].fillna(0).to_numpy()
            fsp = np.hypot(fvx, fvy)
            valid = fsp > 0.5
            if valid.sum() >= 2:
                angles = np.arctan2(fvy[valid], fvx[valid])
                mean_cos = float(np.cos(angles).mean())
                mean_sin = float(np.sin(angles).mean())
                ball_dir_stability = float(np.sqrt(mean_cos ** 2 + mean_sin ** 2))
            else:
                ball_dir_stability = 0.0
        else:
            ball_dir_stability = 0.0

        time_since_prev_loss, time_since_prev_same = BaseEventDataset._previous_loss_time(poss, idx, loss_player)

        if nearest_team_player is not None:
            rcv_vx = float(row.get(f"{nearest_team_player}_vx"))
            rcv_vy = float(row.get(f"{nearest_team_player}_vy"))
            rcv_speed = float(np.hypot(rcv_vx, rcv_vy))
            rcv_x = float(row.get(f"{nearest_team_player}_x"))
            rcv_y = float(row.get(f"{nearest_team_player}_y"))
            if has_direction and rcv_speed > 0.5:
                rcv_align = float((rcv_vx * nx + rcv_vy * ny) / rcv_speed)
            else:
                rcv_align = 0.0
            rcv_pressure = 30.0
            for player in players:
                if player == nearest_team_player or player == loss_player:
                    continue
                if player.startswith(opp_team + "_"):
                    ox = float(row.get(f"{player}_x"))
                    oy = float(row.get(f"{player}_y"))
                    if np.isnan(ox) or np.isnan(oy):
                        continue
                    rcv_pressure = min(rcv_pressure, float(np.hypot(ox - rcv_x, oy - rcv_y)))
        else:
            rcv_speed = 0.0
            rcv_align = 0.0
            rcv_pressure = 30.0

        if len(future) >= 5:
            fx = future["ball_x"].fillna(bx).to_numpy()
            fy = future["ball_y"].fillna(by).to_numpy()
            t_arr = np.arange(len(fx))
            if len(t_arr) > 1:
                ax_, bx_ = np.polyfit(t_arr, fx, 1)
                ay_, by_ = np.polyfit(t_arr, fy, 1)
                rx = fx - (ax_ * t_arr + bx_)
                ry = fy - (ay_ * t_arr + by_)
                traj_curvature = float(np.sqrt(rx ** 2 + ry ** 2).mean())
            else:
                traj_curvature = 0.0
        else:
            traj_curvature = 0.0

        # pd.Na로 기록하면 에러가 발생함.
        sp_type = row.get("set_piece_type", None)
        sp_throw_in = int(sp_type == "ThrowIn")
        sp_corner = int(sp_type in ("CornerKick",))
        sp_freekick = int(sp_type in ("FreeKick", "FreeKick?"))
        sp_goalkick = int(sp_type == "GoalKick")
        sp_kickoff = int(sp_type == "KickOff")
        sp_penalty = int(sp_type == "Penalty")

        ball_x_landing = bx
        ball_y_landing = by
        lands_in_pa = 0
        n_attackers_at_landing = 0
        time_to_landing = 0.0
        shot_on_target_geom = 0

        if include_destination:
            landing_idx = None
            seen_air = False
            for j in range(idx + 1, min(idx + 100, len(poss))):
                zj = float(poss.iloc[j]["ball_z"])
                if zj > 1.5:
                    seen_air = True
                if seen_air and zj < 0.5:
                    landing_idx = j
                    break

            if landing_idx is not None:
                landing_row = poss.iloc[landing_idx]
                ball_x_landing = float(landing_row.get("ball_x"))
                ball_y_landing = float(landing_row.get("ball_y"))
                rel_landing_x = ball_x_landing if loss_team == "home" else PITCH_X - ball_x_landing
                lands_in_pa = int(rel_landing_x >= (PITCH_X - PA_X) and PA_Y_MIN <= ball_y_landing <= PA_Y_MAX)
                n_attackers_at_landing = 0
                for player in players:
                    if not player.startswith(loss_team + "_"):
                        continue
                    px = float(landing_row.get(f"{player}_x"))
                    py = float(landing_row.get(f"{player}_y"))
                    if np.isnan(px) or np.isnan(py):
                        continue
                    rel_px = px if loss_team == "home" else PITCH_X - px
                    if rel_px >= (PITCH_X - PA_X) and PA_Y_MIN <= py <= PA_Y_MAX:
                        n_attackers_at_landing += 1
                time_to_landing = float((landing_idx - idx) / TRACKING_FPS)

            if abs(bvx) > 1e-3:
                t_cross = (gx - bx) / bvx
                if t_cross > 0:
                    y_cross = by + bvy * t_cross
                    if abs(y_cross - PITCH_Y / 2) < GOAL_HALF_W:
                        shot_on_target_geom = 1

        if len(future) >= 25:
            f25 = future.iloc[24]
            ball_x_p1s = float(f25.get("ball_x"))
            ball_y_p1s = float(f25.get("ball_y"))
        else:
            ball_x_p1s = bx
            ball_y_p1s = by
        dist_to_goal_p1s = float(np.hypot(ball_x_p1s - gx, ball_y_p1s - gy))
        delta_dist_to_goal = dist_to_goal_p1s - dist_to_goal

        gain_future = future[future["is_gain"] == True] if "is_gain" in future.columns else pd.DataFrame()
        if len(gain_future) > 0:
            gain_row = gain_future.iloc[0]
            gain_team = gain_row.get("gain_team")
            gain_player = gain_row.get("gain_player")
            time_to_gain = float((gain_row["frame_id"] - row["frame_id"]) / TRACKING_FPS)
            if isinstance(gain_player, str):
                gp_x = float(gain_row.get(f"{gain_player}_x"))
                gp_y = float(gain_row.get(f"{gain_player}_y"))
                gp_rel_x = gp_x if loss_team == "home" else PITCH_X - gp_x
                gain_in_attacking_pa = int(gp_rel_x >= (PITCH_X - PA_X) and PA_Y_MIN <= gp_y <= PA_Y_MAX)
            else:
                gain_in_attacking_pa = 0
            same_team_gain = int(gain_team == loss_team)
            opp_team_gain = int(gain_team == opp_team)
        else:
            time_to_gain = 5.0
            gain_in_attacking_pa = 0
            same_team_gain = 0
            opp_team_gain = 0

        feat = {
            "frame_id": int(row["frame_id"]),
            "period_id": int(row["period_id"]),
            "timestamp": float(row["timestamp"]),
            "loss_player": loss_player,
            "loss_team": loss_team,
            "ball_x": bx,
            "ball_y": by,
            "ball_z": bz,
            "ball_vx": bvx,
            "ball_vy": bvy,
            "ball_speed": bspeed,
            "ball_accel": baccel,
            "ball_z_max_1s": ball_z_max_1s,
            "past_speed_mean": past_speed_mean,
            "past_speed_max": past_speed_max,
            "past_accel_max": past_accel_max,
            "past_z_max": past_z_max,
            "future_speed_max": future_speed_max,
            "future_z_max_2s": future_z_max_2s,
            "future_air_frames": future_air_frames,
            "ball_dir_stability": ball_dir_stability,
            "time_since_prev_loss": time_since_prev_loss,
            "time_since_prev_same": time_since_prev_same,
            "rcv_speed": rcv_speed,
            "rcv_align": rcv_align,
            "rcv_pressure": rcv_pressure,
            "traj_curvature": traj_curvature,
            "sp_throw_in": sp_throw_in,
            "sp_corner": sp_corner,
            "sp_freekick": sp_freekick,
            "sp_goalkick": sp_goalkick,
            "sp_kickoff": sp_kickoff,
            "sp_penalty": sp_penalty,
            "delta_dist_to_goal_1s": delta_dist_to_goal,
            "dist_to_goal": dist_to_goal,
            "cos_to_goal": cos_to_goal,
            "sideline_dist": sideline_dist,
            "rel_x": rel_x,
            "in_attacking_half": int(in_attacking_half),
            "in_shot_zone": int(in_shot_zone),
            "in_cross_zone": int(in_cross_zone),
            "in_attacking_pa": int(in_attacking_pa),
            "lp_x": lp_x,
            "lp_y": lp_y,
            "lp_speed": lp_speed,
            "nearest_opp_dist": nearest_opp_dist,
            "n_opps_in_5m": n_opps_in_5m,
            "n_attackers_in_pa": n_attackers_in_pa,
            "team_in_cone15": team_in_cone15,
            "team_in_cone30": team_in_cone30,
            "opp_in_cone30": opp_in_cone30,
            "nearest_team_along": nearest_team_along,
            "nearest_opp_along": nearest_opp_along,
            "time_to_gain": time_to_gain,
            "gain_in_attacking_pa": int(gain_in_attacking_pa),
            "same_team_gain": same_team_gain,
            "opp_team_gain": opp_team_gain,
            "is_set_piece": int(pd.notna(row.get("set_piece_type", None))),
        }

        if include_destination:
            feat.update(
                {
                    "ball_x_landing": ball_x_landing,
                    "ball_y_landing": ball_y_landing,
                    "lands_in_pa": lands_in_pa,
                    "n_attackers_at_landing": n_attackers_at_landing,
                    "time_to_landing": time_to_landing,
                    "shot_on_target_geom": shot_on_target_geom,
                }
            )

        return feat

    @staticmethod
    def _previous_loss_time(poss: pd.DataFrame, idx: int, loss_player: object) -> tuple[float, float]:
        prev_losses = poss.iloc[:idx]
        prev_any_loss = prev_losses[prev_losses["is_loss"]]
        if len(prev_any_loss) > 0:
            time_since_prev_loss = float((poss.iloc[idx]["frame_id"] - prev_any_loss.iloc[-1]["frame_id"]) / TRACKING_FPS)
        else:
            time_since_prev_loss = 30.0

        prev_same_player = prev_any_loss[prev_any_loss["loss_player"] == loss_player]
        if len(prev_same_player) > 0:
            time_since_prev_same = float((poss.iloc[idx]["frame_id"] - prev_same_player.iloc[-1]["frame_id"]) / TRACKING_FPS)
        else:
            time_since_prev_same = 30.0
        return time_since_prev_loss, time_since_prev_same

class KickDataset(BaseEventDataset):
    task_name = "kick"
    
    def __init__(self, data_path: Path | None = None, cache_path: Path | None = None, save_path: Path | None = None) -> None:
        super().__init__(data_path=data_path, cache_path=cache_path, save_path=save_path)

    def compute_features_and_labels(self, match_id: str) -> pd.DataFrame:
        possession, set_piece, _ = self.detect_frames(match_id) # Detect frames on the fly to ensure we have all the necessary columns for feature extraction
        raw_gt = pd.read_parquet(os.path.join(self.raw_data_path, f"{match_id}.parquet")) # Load raw_gt for label assignment 
  
        # Map set piece types to possession frames
        set_piece_indexed = set_piece.drop_duplicates(subset="frame_id", keep="first").set_index("frame_id")["set_piece_type"]
        possession["set_piece_type"] = possession["frame_id"].map(set_piece_indexed)

        # Identify player columns (e.g., "home_10_x", "away_5_x" -> "home_10", "away_5")
        player_cols = sorted([col[:-2] for col in possession.columns if col.endswith("_x") and (col.startswith("home_") or col.startswith("away_"))])

        feats: list[dict[str, Any]] = []
        valid_loss_cond = (possession["is_loss"] == True) & (possession["ball_state"] == "alive") & possession["loss_player"].notna()
        for loss_idx in possession[valid_loss_cond].index.tolist():
            feat = self.extract_features(possession, loss_idx, player_cols, include_destination=False)
            feat["match_id"] = match_id
            feat["player_id"] = feat.get("loss_player")
            feat["timestamp_s"] = float(possession.iloc[loss_idx]["timestamp"])
            feats.append(feat)

        labels, oba_neg = self.assign_labels(feats, raw_gt, goalkeepers=[], window_sec = 2.0)
        for feat, label, oba in zip(feats, labels, oba_neg):
            feat["label"] = label
            feat["oba_negative"] = oba

        return pd.DataFrame(feats)

    @staticmethod
    def assign_labels(loss_features: list[dict[str, Any]], raw_gt: pd.DataFrame, goalkeepers: list[str], window_sec: float = 2.0) -> tuple[list[str], list[int]]:
        n = len(loss_features)
        labels = ["none"] * n
        oba_neg = [0] * n
        claimed = [False] * n

        idx_by_key: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
        for i, feat in enumerate(loss_features):
            player = feat.get("player_id") or feat.get("loss_player")
            if isinstance(player, str):
                idx_by_key[(int(feat["period_id"]), player)].append((float(feat["timestamp"]), i))

        events: list[tuple[int, float, str, str]] = []
        for _, row in raw_gt.iterrows():
            kind = row["event_kind"]
            player = row.get("player_id")
            if not isinstance(player, str):
                continue
            if kind == "Pass":
                target = "pass"
            elif kind == "Cross":
                target = "cross"
            elif kind == "ShotAtGoal":
                target = "shot"
            elif kind == "OtherBallAction":
                if str(row.get("defensive_clearance")).lower() == "true":
                    target = "pass"
                else:
                    # OtherBallAction는 clearance, bad_touch, miss 등 다양한 행동이 섞여있는데, 우선 명확히 기록이 된 clearance만 pass로 간주하고 나머지는 무시
                    ev_ts = row.get("sync_ts", row["timestamp"])
                    if not np.isfinite(ev_ts):
                        ev_ts = row["timestamp"]
                    events.append((int(row["period_id"]), float(ev_ts), player, "_oba_neg"))
                    continue
            else:
                continue

            ev_ts = row.get("sync_ts", row["timestamp"])
            if not np.isfinite(ev_ts):
                ev_ts = row["timestamp"]
            events.append((int(row["period_id"]), float(ev_ts), player, target))

        for period, ev_ts, player, target in events:
            bucket = idx_by_key.get((period, player), [])
            best_i, best_dt = -1, window_sec + 1.0
            for ts, i in bucket:
                if claimed[i]:
                    continue
                dt = abs(ts - ev_ts)
                if dt <= window_sec and dt < best_dt:
                    best_dt = dt
                    best_i = i
            if best_i < 0:
                continue
            if target == "_oba_neg": # OtherBallAction 유형이 예측이 어렵기 때문에 negative weight를 부여하기 위해 별도의 라벨로 처리
                oba_neg[best_i] = 1
            else:
                labels[best_i] = target
                claimed[best_i] = True

        return labels, oba_neg
