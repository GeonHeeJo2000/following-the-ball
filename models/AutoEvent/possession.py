import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# import tools.config as cfg
from models.AutoEvent import config as cfg

from models.AutoEvent.helpers import get_players


# ADAPTIVE_MODE: "off" | "full" | "pass_only"
#   off       → no intermediate loss (baseline)
#   full      → fill all gaps ≥ MIN_GAP_FRAMES
#   pass_only → skip gaps where ball is in shot/cross zone (rel_x ≥ 75m)
ADAPTIVE_MODE = os.environ.get("ADAPTIVE_MODE", "full").lower()


class PossessionDetector:
    def __init__(self, tracking: pd.DataFrame, pre_smoothed: bool = False):
        self.tracking = tracking.copy()
        self.players = get_players(self.tracking)
        self._pre_smoothed = pre_smoothed

    def run(self):
        if not self._pre_smoothed:
            self.smooth_ball()
        return (
            self.add_ball_kinematics()
            .add_player_ball_distances()
            .add_ball_control()
            .add_control_sequences()
            .add_possession_losses()
            .add_duel_losses()
            .add_possession_gains()
            .add_adaptive_losses()
            .cleanup_dribble_losses()
            # Zone-specific recovery for rare events that the generic
            # adaptive detector misses because they happen mid-possession
            # (first-time strikes, headers, volleys). These rules are
            # highly specific so they add few false positives.
            .add_shot_losses()
            # .filter_aerial_losses()
            .tracking
        )

    # Savitzy-Golay filter로 볼 위치 smoothing (이후 속도 계산에 사용)
    def smooth_ball(self):
        alive_mask = self.tracking["ball_state"].eq("alive")
        self.tracking.loc[~alive_mask, ["ball_x", "ball_y"]] = np.nan

        # alive / dead가 바뀔 때마다 새로운 구간 id 부여
        episode_id = alive_mask.ne(alive_mask.shift(fill_value=False)).cumsum()

        for seg in episode_id[alive_mask].unique():
            idx = self.tracking.index[(episode_id == seg) & alive_mask]
            xy = self.tracking.loc[idx, ["ball_x", "ball_y"]]

            if len(xy) == 0:
                continue

            window = min(cfg.SG_WINDOW, len(xy))
            if window % 2 == 0:
                window -= 1

            # 너무 짧은 구간은 smoothing하지 않고 그대로 둠
            if window < 3 or len(xy) < cfg.SG_POLYORDER + 2:
                continue

            self.tracking.loc[idx, "ball_x"] = savgol_filter(
                xy["ball_x"].to_numpy(),
                window_length=window,
                polyorder=min(cfg.SG_POLYORDER, window - 1),
                mode="interp",
            )
            self.tracking.loc[idx, "ball_y"] = savgol_filter(
                xy["ball_y"].to_numpy(),
                window_length=window,
                polyorder=min(cfg.SG_POLYORDER, window - 1),
                mode="interp",
            )

        return self    

    def add_ball_kinematics(self):

        self.tracking["ball_dx_prev"] = self.tracking["ball_x"] - self.tracking["ball_x"].shift(1)
        self.tracking["ball_dy_prev"] = self.tracking["ball_y"] - self.tracking["ball_y"].shift(1)

        self.tracking["ball_dx_next"] = self.tracking["ball_x"].shift(-1) - self.tracking["ball_x"]
        self.tracking["ball_dy_next"] = self.tracking["ball_y"].shift(-1) - self.tracking["ball_y"]

        self.tracking["ball_speed_prev"] = np.sqrt(
            self.tracking["ball_dx_prev"] ** 2 + self.tracking["ball_dy_prev"] ** 2
        )
        self.tracking["ball_speed_next"] = np.sqrt(
            self.tracking["ball_dx_next"] ** 2 + self.tracking["ball_dy_next"] ** 2
        )

        prev_nonzero = self.tracking["ball_speed_prev"].replace(0, np.nan)
        next_nonzero = self.tracking["ball_speed_next"].replace(0, np.nan)

        # 공의 이동 방향 (단위 벡터)
        self.tracking["ball_dir_in_x"] = self.tracking["ball_dx_prev"] / prev_nonzero
        self.tracking["ball_dir_in_y"] = self.tracking["ball_dy_prev"] / prev_nonzero

        self.tracking["ball_dir_out_x"] = self.tracking["ball_dx_next"] / next_nonzero
        self.tracking["ball_dir_out_y"] = self.tracking["ball_dy_next"] / next_nonzero

        # 공의 이동량
        self.tracking["ball_displacement"] = self.tracking["ball_speed_next"]

        return self
    
    def add_player_ball_distances(self):
        for player in self.players:
            self.tracking[f"dist_{player}"] = np.sqrt(
                (self.tracking[f"{player}_x"] - self.tracking["ball_x"]) ** 2
                + (self.tracking[f"{player}_y"] - self.tracking["ball_y"]) ** 2
            )
        return self

    def add_ball_control(self):
        # None으로 기록하는 이유: pd.Na 포맷으로 parquet로 저장하면 None으로 타입이 변환되기 때문에 처음부터 None으로 초기화
        self.tracking["ball_control"] = "no_possession"
        self.tracking["controller_id"] = None
        self.tracking["controller_team"] = None
        self.tracking["controller_distance"] = None
        self.tracking["duel_players"] = None

        for idx, row in self.tracking.iterrows():
            if row["ball_state"] == "dead":
                self.tracking.at[idx, "ball_control"] = "dead_ball"
                continue

            home_in_pz, away_in_pz = [], []
            home_in_dz, away_in_dz = [], []

            for player in self.players:
                dist = row[f"dist_{player}"]
                if pd.isna(dist):
                    continue

                is_home = player.startswith("home_")

                if dist <= cfg.R_PZ:
                    (home_in_pz if is_home else away_in_pz).append((player, dist))

                if dist <= cfg.R_DZ:
                    (home_in_dz if is_home else away_in_dz).append((player, dist))

            if home_in_dz and away_in_dz:
                duel_candidates = sorted(home_in_dz + away_in_dz, key=lambda x: x[1])
                self.tracking.at[idx, "ball_control"] = "duel"
                self.tracking.at[idx, "duel_players"] = "|".join(player for player, _ in duel_candidates)
                continue

            possession_candidates = home_in_pz + away_in_pz
            if possession_candidates:
                controller_id, controller_distance = min(possession_candidates, key=lambda x: x[1])
                self.tracking.at[idx, "ball_control"] = "possession"
                self.tracking.at[idx, "controller_id"] = controller_id
                self.tracking.at[idx, "controller_team"] = controller_id.split("_")[0]
                self.tracking.at[idx, "controller_distance"] = controller_distance

        return self

    def add_control_sequences(self):
        self.tracking["control_sequence_id"] = None
        self.tracking["control_sequence_type"] = None
        self.tracking["control_sequence_player"] = None

        seq_id = 0
        prev_type = None
        prev_player = None

        for idx, row in self.tracking.iterrows():
            cur_type = row["ball_control"]

            if cur_type not in ["possession", "duel"]:
                prev_type = None
                prev_player = None
                continue

            if cur_type == "possession":
                cur_player = row["controller_id"]
                same_sequence = (prev_type == "possession") and (prev_player == cur_player)
            else:
                cur_player = None
                same_sequence = (prev_type == "duel")

            if not same_sequence:
                seq_id += 1

            self.tracking.at[idx, "control_sequence_id"] = seq_id
            self.tracking.at[idx, "control_sequence_type"] = cur_type
            self.tracking.at[idx, "control_sequence_player"] = cur_player

            prev_type = cur_type
            prev_player = cur_player

        return self
    
    def add_possession_losses(self):
        self.tracking["is_loss"] = False
        self.tracking["loss_player"] = None
        self.tracking["loss_team"] = None

        # control_sequence 단위로 처리: 각 시퀀스의 마지막 프레임만 loss 후보
        sequence_ids = self.tracking["control_sequence_id"].dropna().unique()

        for seq_id in sequence_ids:
            seq_mask = self.tracking["control_sequence_id"] == seq_id
            seq = self.tracking[seq_mask]

            if seq.empty:
                continue
            if seq["control_sequence_type"].iloc[0] != "possession":
                continue

            player = seq["control_sequence_player"].iloc[0]
            if player is None:
                continue

            # 시퀀스의 마지막 프레임이 loss 후보
            last_idx = seq.index[-1]
            i = self.tracking.index.get_loc(last_idx)

            if i + 1 >= len(self.tracking):
                continue

            row = self.tracking.iloc[i]
            ball_displacement = row["ball_displacement"]
            next_row = self.tracking.iloc[i + 1]
            next_dist = next_row[f"dist_{player}"]
            next_control = next_row["ball_control"]

            outside_pz = pd.isna(next_dist) or (next_dist > cfg.R_PZ)
            # Sequence also legitimately ends when next frame becomes a
            # duel (contested) or no_possession (ball loose), even if the
            # player is still physically inside R_PZ.
            sequence_ended = (
                outside_pz
                or next_control in ("duel", "no_possession")
                or (next_control == "possession"
                    and next_row.get("controller_id") != player)
            )
            enough_movement = (not pd.isna(ball_displacement)) and (ball_displacement > cfg.EPS_S)

            if not (sequence_ended and enough_movement):
                continue

            next_control_idx = None
            for j in range(i + 1, len(self.tracking)):
                if self.tracking.iloc[j]["ball_control"] in ["possession", "duel"]:
                    next_control_idx = j
                    break

            if next_control_idx is None:
                self.tracking.at[last_idx, "is_loss"] = True
                self.tracking.at[last_idx, "loss_player"] = player
                self.tracking.at[last_idx, "loss_team"] = player.split("_")[0]
                continue

            next_row = self.tracking.iloc[next_control_idx]

            if next_row["ball_control"] == "possession" and next_row["controller_id"] == player:
                # i+1 ~ next_control_idx 사이에 dead ball이 있으면 loss
                between = self.tracking.iloc[i + 1 : next_control_idx]
                if not between.empty and (between["ball_state"] == "dead").any():
                    pass
                else:
                    continue

            self.tracking.at[last_idx, "is_loss"] = True
            self.tracking.at[last_idx, "loss_player"] = player
            self.tracking.at[last_idx, "loss_team"] = player.split("_")[0]

        return self
    
    def add_possession_gains(self):
        """
        gain 규칙:
        같은 선수의 control sequence [f0, ..., fn]에 대해
        - 시작의 incoming direction과 끝의 outgoing direction이 충분히 다르거나
        - sequence 내부 어떤 frame에서든 speed 변화가 충분히 크면
        gain을 sequence 시작점 f0에 기록
        """
                
        self.tracking["is_gain"] = False
        self.tracking["gain_player"] = None
        self.tracking["gain_team"] = None

        sequence_ids = self.tracking["control_sequence_id"].dropna().unique()

        for seq_id in sequence_ids:
            seq = self.tracking[self.tracking["control_sequence_id"] == seq_id]

            if seq.empty:
                continue

            if seq["control_sequence_type"].iloc[0] != "possession":
                continue

            player = seq["control_sequence_player"].iloc[0]
            if player is None:
                continue

            start_idx = seq.index[0]
            end_idx = seq.index[-1]

            start_row = self.tracking.loc[start_idx]
            end_row = self.tracking.loc[end_idx]

            has_direction_change = False
            has_speed_change = False

            if not (
                pd.isna(start_row["ball_dir_in_x"]) or
                pd.isna(start_row["ball_dir_in_y"]) or
                pd.isna(end_row["ball_dir_out_x"]) or
                pd.isna(end_row["ball_dir_out_y"])
            ):
                dot_product = (
                    start_row["ball_dir_in_x"] * end_row["ball_dir_out_x"] +
                    start_row["ball_dir_in_y"] * end_row["ball_dir_out_y"]
                )
                if dot_product < cfg.EPS_THETA:
                    has_direction_change = True

            valid_speed = seq[["ball_speed_prev", "ball_speed_next"]].dropna()
            if not valid_speed.empty:
                speed_diff = np.abs(
                    valid_speed["ball_speed_next"] - valid_speed["ball_speed_prev"]
                )
                if np.any(speed_diff > cfg.EPS_V):
                    has_speed_change = True

            if has_direction_change or has_speed_change:
                self.tracking.at[start_idx, "is_gain"] = True
                self.tracking.at[start_idx, "gain_player"] = player
                self.tracking.at[start_idx, "gain_team"] = player.split("_")[0]

        # 중간에 loss 없는 연속 same-player gain 제거
        gain_idx_list = self.tracking.index[self.tracking["is_gain"]].tolist()
        last_gain: dict = {}  # player -> positional index
        for label_idx in gain_idx_list:
            pos = self.tracking.index.get_loc(label_idx)
            player = self.tracking.at[label_idx, "gain_player"]
            if player is None:
                continue
            if player in last_gain:
                prev_pos = last_gain[player]
                between = self.tracking.iloc[prev_pos + 1 : pos]
                # 본인 loss OR 다른 선수의 gain(= 볼이 다른 선수에게 넘어갔음)이 있으면 허용
                has_loss = (between["is_loss"] & (between["loss_player"] == player)).any()
                other_gained = (between["is_gain"] & (between["gain_player"] != player)).any()
                if not has_loss and not other_gained:
                    self.tracking.at[label_idx, "is_gain"] = False
                    self.tracking.at[label_idx, "gain_player"] = None
                    self.tracking.at[label_idx, "gain_team"] = None
                    continue
            last_gain[player] = pos

        return self

    def add_adaptive_losses(self):
        """Gap-aware intermediate loss detection.

        Identify time gaps between consecutive losses (within same period).
        In each gap longer than MIN_GAP_FRAMES, find ball acceleration peaks
        (kicks/deflections) and add them as additional loss frames.

        Player attribution: closest player to ball at the peak frame,
        within 3m to avoid attributing to far-away players.

        Mode is controlled by ADAPTIVE_MODE env var (off / full / pass_only).
        """
        if ADAPTIVE_MODE == "off":
            return self

        # Original behavior: only fire in long (≥3s) gaps with low
        # threshold. Specialized recovery for rare events (shot/cross)
        # is now handled by zone-specific add_shot_losses/add_cross_losses
        # that run after cleanup_dribble_losses.
        MIN_GAP_FRAMES = 75         # 3 s @ 25fps — only fill gaps ≥ 3 s
        ACCEL_THRESH = 5.0          # m/s² — kick-like acceleration
        MAX_PLAYER_DIST = 3.0       # m — max distance for attribution
        MIN_PEAK_SEPARATION = 25    # 1 s — peaks must be 1 s apart
        # pass_only: skip if ball is in shot/cross zone (rel_x ≥ 75m)
        PASS_ONLY = (ADAPTIVE_MODE == "pass_only")

        # Collect current loss frames per period (sorted by frame_id)
        loss_by_period = {}
        for label_idx in self.tracking.index[self.tracking["is_loss"]]:
            pid = int(self.tracking.at[label_idx, "period_id"])
            pos = self.tracking.index.get_loc(label_idx)
            loss_by_period.setdefault(pid, []).append(pos)
        for pid in loss_by_period:
            loss_by_period[pid].sort()

        added = 0
        for pid, positions in loss_by_period.items():
            if not positions:
                continue
            # Iterate consecutive loss pairs (and edges)
            period_frames = self.tracking.index[self.tracking["period_id"] == pid]
            if len(period_frames) == 0:
                continue
            first_pos = self.tracking.index.get_loc(period_frames[0])
            last_pos = self.tracking.index.get_loc(period_frames[-1])
            # Edge gaps: before first loss, after last loss
            sentinels = [first_pos - 1] + positions + [last_pos + 1]
            for i in range(len(sentinels) - 1):
                gap_start = sentinels[i] + 1
                gap_end = sentinels[i + 1] - 1
                if gap_end - gap_start + 1 < MIN_GAP_FRAMES:
                    continue
                accel_thresh_use = ACCEL_THRESH
                # Find acceleration peaks in this gap (alive frames only)
                gap_slice = self.tracking.iloc[gap_start:gap_end + 1]
                gap_alive = gap_slice[gap_slice["ball_state"] == "alive"]
                if gap_alive.empty:
                    continue
                # Get ball_accel — note: ball_accel might be NaN/missing
                if "ball_accel" not in gap_alive.columns:
                    continue
                accel = gap_alive["ball_accel"].fillna(0.0).abs().to_numpy()
                idx_in_gap = gap_alive.index.tolist()
                if len(accel) < 3:
                    continue

                # Peak detection: local maxima above threshold,
                # with min separation
                peaks = []
                for j in range(1, len(accel) - 1):
                    if accel[j] < accel_thresh_use:
                        continue
                    if accel[j] <= accel[j - 1] or accel[j] <= accel[j + 1]:
                        continue
                    if peaks and (j - peaks[-1]) < MIN_PEAK_SEPARATION:
                        if accel[j] > accel[peaks[-1]]:
                            peaks[-1] = j
                        continue
                    peaks.append(j)

                # Add each peak as a loss
                for pk in peaks:
                    label_idx = idx_in_gap[pk]
                    if self.tracking.at[label_idx, "is_loss"]:
                        continue
                    row = self.tracking.loc[label_idx]
                    # Find closest player to ball at peak
                    bx = row.get("ball_x")
                    by = row.get("ball_y")
                    if pd.isna(bx) or pd.isna(by):
                        continue
                    best_player = None
                    best_dist = float("inf")
                    for player in self.players:
                        d = row.get(f"dist_{player}")
                        if d is None:
                            continue
                        if d < best_dist:
                            best_dist = d
                            best_player = player
                    if best_player is None or best_dist > MAX_PLAYER_DIST:
                        continue
                    # pass_only: skip if ball is in shot/cross zone of the
                    # candidate player's attacking direction.
                    if PASS_ONLY:
                        team = best_player.split("_")[0]
                        rel_x = bx if team == "home" else cfg.PITCH_X - bx
                        if rel_x >= cfg.SHOT_ZONE_X:  # 75m
                            continue
                    self.tracking.at[label_idx, "is_loss"] = True
                    self.tracking.at[label_idx, "loss_player"] = best_player
                    self.tracking.at[label_idx, "loss_team"] = best_player.split("_")[0]
                    added += 1

        return self

    def add_duel_losses(self):
        """For duel sequences, mark a loss at duel-end.

        A duel = multiple players in R_PZ, no single controller. The
        original add_possession_losses ignores duel sequences entirely.
        But duels often end when one player kicks the ball away — that
        kick is a real "loss" event by that player.

        Heuristic: at the LAST frame of each duel sequence:
          - require ball_displacement > EPS_S (ball is moving)
          - find closest player at that frame
          - if closest player is within R_PZ → mark loss for them
        """
        sequence_ids = (self.tracking["control_sequence_id"]
                        .dropna().unique())
        added = 0
        for seq_id in sequence_ids:
            seq_mask = self.tracking["control_sequence_id"] == seq_id
            seq = self.tracking[seq_mask]
            if seq.empty:
                continue
            if seq["control_sequence_type"].iloc[0] != "duel":
                continue

            last_idx = seq.index[-1]
            i = self.tracking.index.get_loc(last_idx)
            row = self.tracking.iloc[i]
            ball_disp = row["ball_displacement"]
            if pd.isna(ball_disp) or ball_disp <= cfg.EPS_S:
                continue

            bx = row.get("ball_x")
            by = row.get("ball_y")
            if pd.isna(bx) or pd.isna(by):
                continue

            # Find closest player at duel-end
            best_player = None
            best_dist = float("inf")
            for player in self.players:
                d = row.get(f"dist_{player}")
                if d is None:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_player = player
            if best_player is None or best_dist > cfg.R_PZ:
                continue

            # Don't overwrite an existing loss
            if self.tracking.at[last_idx, "is_loss"]:
                continue

            self.tracking.at[last_idx, "is_loss"] = True
            self.tracking.at[last_idx, "loss_player"] = best_player
            self.tracking.at[last_idx, "loss_team"] = (
                best_player.split("_")[0])
            added += 1
        return self

    def cleanup_dribble_losses(self):
        """Collapse multiple loss frames within a single possession episode.

        A possession episode is the interval where one player is the
        "current owner" (defined by the latest gain_player). Within
        this interval, multiple is_loss=True frames attributed to the
        same owner are dribble noise (the player is still in control
        and just touching the ball). Keep only the LAST one — that
        represents the terminal action (pass, shot, etc.) by which
        possession is released.

        Loss frames attributed to a DIFFERENT player from the current
        owner (e.g., defender deflection) are kept as-is.
        """
        # Iterate in tracking order (already sorted by frame_id)
        idx = self.tracking.index.tolist()
        is_loss_arr = self.tracking["is_loss"].to_numpy()
        is_gain_arr = self.tracking["is_gain"].to_numpy()
        loss_player_arr = self.tracking["loss_player"].to_numpy()
        gain_player_arr = self.tracking["gain_player"].to_numpy()
        period_arr = self.tracking["period_id"].to_numpy()

        keep = is_loss_arr.copy()
        n = len(idx)
        # Group same-period episodes: episode = consecutive frames with
        # same current owner. Reset owner at period boundary.
        current_owner = None
        current_period = None
        same_owner_losses: list[int] = []  # indices

        def flush(losses: list[int]):
            # Drop all but last
            if len(losses) > 1:
                for k in losses[:-1]:
                    keep[k] = False

        for i in range(n):
            # Period change → flush previous, reset
            if period_arr[i] != current_period:
                flush(same_owner_losses)
                same_owner_losses = []
                current_period = period_arr[i]
                current_owner = None
            # Gain by a player → set/change owner
            if is_gain_arr[i] and isinstance(gain_player_arr[i], str):
                if gain_player_arr[i] != current_owner:
                    flush(same_owner_losses)
                    same_owner_losses = []
                    current_owner = gain_player_arr[i]
            # Loss by current owner → add to same_owner_losses
            if (is_loss_arr[i] and isinstance(loss_player_arr[i], str)
                    and loss_player_arr[i] == current_owner):
                same_owner_losses.append(i)
        flush(same_owner_losses)

        n_before = int(is_loss_arr.sum())
        n_after = int(keep.sum())
        print(f"  cleanup_dribble_losses: {n_before} → {n_after} "
              f"({n_before - n_after} dropped)")

        # Apply
        self.tracking["is_loss"] = keep
        drop_mask = is_loss_arr & ~keep
        drop_idx = self.tracking.index[drop_mask]
        self.tracking.loc[drop_idx, "loss_player"] = None
        self.tracking.loc[drop_idx, "loss_team"] = None
        return self

    def add_shot_losses(self):
        """Zone-specific shot recovery (multi-tier).

        Three tiers progressively widen detection while staying highly
        specific to shot signatures:

        - Tier 1 (standard): the kinematic signature most shots have —
          strong kick (accel ≥ 25 m/s²), in attacking third near goal
          (rel_x ≥ 70m, d_goal ≤ 35m), ball heading toward goal
          (cos ≥ 0.2).
        - Tier 2 (close-range): inside-the-box shots where ball
          direction can be erratic (headers, deflections, sliding
          finishes). d_goal ≤ 12m + rel_x ≥ 95m, very loose direction.
        - Tier 3 (override): strong shot signature even if Stage 2
          already attributed the loss to a defender/keeper. d_goal ≤ 18m
          + accel ≥ 50 + cos ≥ 0 + same-team shooter very close.

        Statistics on 225 GT shots (median / p10):
            accel = 84 / 28,  dist_to_goal = 17 / 7,
            cos_to_goal = 0.95 / 0.08,  rel_x = 91 / 81.
        """
        # Tier 1: standard shot signature
        SHOT_ACCEL = 25.0
        SHOT_REL_X = 70.0
        SHOT_GOAL_DIST_MAX = 35.0
        SHOT_GOAL_COS = 0.2
        # Tier 2: close-range loose-direction (headers, tap-ins, deflections,
        # rebounds — at close range ball direction can be anything).
        CLOSE_GOAL_DIST = 15.0      # box-edge generous
        CLOSE_REL_X = 90.0
        CLOSE_ACCEL = 5.0           # tap-ins, weak touches
        CLOSE_COS = -1.0            # any direction (rebounds, deflections)
        # Tier 3: high-confidence override of existing is_loss
        OVERRIDE_GOAL_DIST = 18.0
        OVERRIDE_REL_X = 90.0
        OVERRIDE_ACCEL = 50.0
        OVERRIDE_COS = -0.3         # allow deflected shots (cos negative)
        OVERRIDE_MAX_DIST = 3.0
        # Shared
        MAX_PLAYER_DIST = 5.0       # catches d_shooter=3-5m cases
        MIN_SEPARATION = 5          # 0.2 s — allows nearby double-touches

        bx = self.tracking["ball_x"].to_numpy()
        by = self.tracking["ball_y"].to_numpy()
        bvx = self.tracking["ball_vx"].fillna(0.0).to_numpy()
        bvy = self.tracking["ball_vy"].fillna(0.0).to_numpy()
        accel = self.tracking["ball_accel"].fillna(0.0).abs().to_numpy()
        is_loss = self.tracking["is_loss"].to_numpy()
        state = self.tracking["ball_state"].to_numpy()

        # Goal positions: home defends x=0, attacks x=PITCH_X. Vice versa.
        goal_home_attack = (cfg.PITCH_X, cfg.PITCH_Y / 2)
        goal_away_attack = (0.0, cfg.PITCH_Y / 2)

        def classify_tier(i, d, cos, rel_x, accel_val, override_capable):
            """Return tier name (str) or None if no shot signature."""
            # Tier 1: standard signature
            if (accel_val >= SHOT_ACCEL and d <= SHOT_GOAL_DIST_MAX
                    and rel_x >= SHOT_REL_X and cos >= SHOT_GOAL_COS):
                return "std"
            # Tier 2: close-range loose-direction
            if (d <= CLOSE_GOAL_DIST and rel_x >= CLOSE_REL_X
                    and accel_val >= CLOSE_ACCEL and cos >= CLOSE_COS):
                return "close"
            # Tier 3: high-confidence override (used if frame already is_loss)
            if (override_capable and d <= OVERRIDE_GOAL_DIST
                    and rel_x >= OVERRIDE_REL_X
                    and accel_val >= OVERRIDE_ACCEL
                    and cos >= OVERRIDE_COS):
                return "override"
            return None

        def compute_sig(i, attacking_x_dir):
            """For a hypothesized attacking direction, return (d_goal,
            cos_goal, rel_x) of the ball at frame i."""
            goal = goal_home_attack if attacking_x_dir > 0 else goal_away_attack
            gx, gy = goal
            d_goal = float(np.hypot(gx - bx[i], gy - by[i]))
            v_norm = float(np.hypot(bvx[i], bvy[i])) + 1e-6
            cos_goal = ((gx - bx[i]) * bvx[i]
                        + (gy - by[i]) * bvy[i]) / (d_goal * v_norm)
            rel_x = bx[i] if attacking_x_dir > 0 else cfg.PITCH_X - bx[i]
            return d_goal, cos_goal, rel_x

        def closest_player_on_team(i, team_prefix, max_dist):
            """Closest player on the named team within max_dist of ball."""
            row = self.tracking.iloc[i]
            best_player = None
            best_dist = float("inf")
            for player in self.players:
                if not player.startswith(team_prefix + "_"):
                    continue
                d = row.get(f"dist_{player}")
                if d is None:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_player = player
            if best_player is None or best_dist > max_dist:
                return None, float("inf")
            return best_player, best_dist

        added = 0
        overridden = 0
        last_added = -MIN_SEPARATION - 1
        for i in range(1, len(accel) - 1):
            if state[i] != "alive":
                continue
            if accel[i] < CLOSE_ACCEL:   # earliest cutoff for any tier
                continue
            # Local max
            if accel[i] <= accel[i - 1] or accel[i] <= accel[i + 1]:
                continue
            # Cooldown
            if i - last_added < MIN_SEPARATION:
                continue
            # Try BOTH attacking directions, pick whichever team has a
            # valid shot signature + a close attacking-team player.
            # This avoids GK/defender misattribution where the closest
            # player to ball isn't on the attacking team.
            best_match = None  # (tier, player, team, dist)
            for attacking_team, attacking_dir in (("home", +1), ("away", -1)):
                d_g, cos_g, rx = compute_sig(i, attacking_dir)
                max_dist = OVERRIDE_MAX_DIST if is_loss[i] else MAX_PLAYER_DIST
                player, dist = closest_player_on_team(
                    i, attacking_team, max_dist)
                if player is None:
                    continue
                tier = classify_tier(i, d_g, cos_g, rx, accel[i],
                                     override_capable=is_loss[i])
                if tier is None:
                    continue
                if is_loss[i] and tier != "override":
                    continue
                # Prefer closer player on stronger tier
                tier_rank = {"std": 2, "close": 1, "override": 3}
                score = (tier_rank[tier], -dist)
                if best_match is None or score > best_match[0]:
                    best_match = (score, player, attacking_team, dist)
            if best_match is None:
                continue
            _, best_player, team, _ = best_match

            # Insert (or override)
            label_idx = self.tracking.index[i]
            was_loss = is_loss[i]
            self.tracking.at[label_idx, "is_loss"] = True
            self.tracking.at[label_idx, "loss_player"] = best_player
            self.tracking.at[label_idx, "loss_team"] = team
            last_added = i
            if was_loss:
                overridden += 1
            else:
                added += 1

        print(f"  add_shot_losses: added {added} new + {overridden} "
              f"overridden = {added + overridden} shot-specific frames")
        return self

    def filter_aerial_losses(self, z_thresh: float = 3.0):
        """ball_z가 z_thresh 이상인 프레임의 loss를 제거한다."""
        if "ball_z" not in self.tracking.columns:
            return self
        aerial_loss = (self.tracking["ball_z"] >= z_thresh) & self.tracking["is_loss"]
        n = int(aerial_loss.sum())
        if n:
            self.tracking.loc[aerial_loss, "is_loss"] = False
            self.tracking.loc[aerial_loss, "loss_player"] = None
            self.tracking.loc[aerial_loss, "loss_team"] = None
            print(f"  filter_aerial_losses: removed {n} losses (ball_z ≥ {z_thresh})")
        return self

    def add_intermediate_losses(self):
        """Detect 'kick events' within possession sequences and add as additional
        loss frames. A possession sequence by player P may contain a brief kick
        where the ball is released and quickly returns (player retained). Such
        intermediate kicks should also be counted as 'pass-like' loss events.

        Criteria for an intermediate loss at frame f:
          - f is inside a possession sequence (not start, not end)
          - sharp ball direction change at f (cos(angle(in, out)) < 0.94 ≈ 20°)
          - significant speed change at f (|next_speed - prev_speed| > 0.3 m/frame)
          - ball is moving (ball_speed_next > 0.2 m/frame)
        """
        # Direction change cosine threshold (cos 20°)
        DIR_THRESH = 0.94
        SPEED_DIFF_THRESH = 0.3   # m/frame
        MIN_BALL_SPEED = 0.2      # m/frame

        sequence_ids = self.tracking["control_sequence_id"].dropna().unique()
        added = 0

        for seq_id in sequence_ids:
            seq_mask = self.tracking["control_sequence_id"] == seq_id
            seq = self.tracking[seq_mask]
            if seq.empty:
                continue
            if seq["control_sequence_type"].iloc[0] != "possession":
                continue
            player = seq["control_sequence_player"].iloc[0]
            if player is None:
                continue

            seq_indices = seq.index.tolist()
            if len(seq_indices) < 5:
                continue

            # Skip first and last (already handled by loss/gain detectors)
            for label_idx in seq_indices[1:-1]:
                row = self.tracking.loc[label_idx]
                if self.tracking.at[label_idx, "is_loss"]:
                    continue  # already a loss
                if self.tracking.at[label_idx, "is_gain"]:
                    continue  # already a gain

                # Direction change check
                ix, iy = row.get("ball_dir_in_x"), row.get("ball_dir_in_y")
                ox, oy = row.get("ball_dir_out_x"), row.get("ball_dir_out_y")
                if pd.isna(ix) or pd.isna(iy) or pd.isna(ox) or pd.isna(oy):
                    continue
                cos_change = ix * ox + iy * oy
                if cos_change >= DIR_THRESH:
                    continue

                # Speed change check
                sp_prev = row.get("ball_speed_prev", 0)
                sp_next = row.get("ball_speed_next", 0)
                if pd.isna(sp_prev) or pd.isna(sp_next):
                    continue
                speed_diff = abs(sp_next - sp_prev)
                if speed_diff <= SPEED_DIFF_THRESH:
                    continue
                if sp_next < MIN_BALL_SPEED:
                    continue

                self.tracking.at[label_idx, "is_loss"] = True
                self.tracking.at[label_idx, "loss_player"] = player
                self.tracking.at[label_idx, "loss_team"] = player.split("_")[0]
                added += 1

        return self