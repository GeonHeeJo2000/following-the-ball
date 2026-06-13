import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
from scipy.signal import find_peaks
from shapely.geometry import LineString, Point, Polygon

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Postprocessor:
    def __init__(self, traces: pd.DataFrame, teams: pd.DataFrame, fps=10, pitch_size: tuple = (105, 68)) -> None:
        self.fps = fps
        self.pitch_size = pitch_size
        self.BALL_x, self.BALL_y, self.BALL_z = "B00_x", "B00_y", "B00_z"
        self.traces = traces
        self.teams = teams
        self.home_gk_players = teams[(teams.position == "GK") & (teams.team_id == "home")].player_code.tolist()
        self.away_gk_players = teams[(teams.position == "GK") & (teams.team_id == "away")].player_code.tolist()

        output_cols = ["carrier", "ball_x", "ball_y", "ball_z", "candidate_carriers"]
        self.output = pd.DataFrame(index=self.traces.index, columns=output_cols)
        self.output[output_cols[1:]] = self.output[output_cols[1:]].astype(float)

        self.control_records_cols = ["control_idx", "player", "type", "subtype", "control_score", "candidate_players"]
        self.event_records_cols = ["start_idx", "end_idx", "player", "receiver", "type", "subtype", "success"]

        # Align Out Pitch Boundary Coordinates
        outside_labels = ["OUT-L", "OUT-R", "OUT-B", "OUT-T"]
        outside_x = [0, pitch_size[0], pitch_size[0] / 2, pitch_size[0] / 2]
        outside_y = [pitch_size[1] / 2, pitch_size[1] / 2, 0, pitch_size[1]]
        for i, label in enumerate(outside_labels):
            self.traces[f"{label}_x"] = float(outside_x[i])
            self.traces[f"{label}_y"] = float(outside_y[i])
            self.traces[[f"{label}_vx", f"{label}_vy", f"{label}_speed", f"{label}_accel"]] = 0
            print(f"label={label}: x={float(outside_x[i])}, y={float(outside_y[i])}")

        # Mapping of player_code to team and period_id for quick lookup
        self.direction_dict = {"Home": {}, "Away": {}}  # Cache for playing direction by team and period
        for period_id, period_tracking in traces.groupby('period_id'):
            print(f"====={period_id}=====")
            self.direction_dict["Home"][period_id] = self.find_playing_direction(period_tracking, "H")
            self.direction_dict["Away"][period_id] = self.find_playing_direction(period_tracking, "A")
        print(f"Playing direction dict: {self.direction_dict}")
        assert all(d in [-1, 1] for team_direction_dict in self.direction_dict.values() for d in team_direction_dict.values()), "Playing directions must be either -1 or 1."
        
    def find_playing_direction(self, traces: pd.DataFrame, team_name: str) -> int:
        """
            첫번째 traces 데이터의 home team이 왼쪽에서 오른쪽으로 공격하는지(1) 또는 오른쪽에서 왼쪽으로 공격하는지(-1)를 판단한다.
        """
        
        first_row = traces.iloc[0].dropna()
        gk_x_columns = [f"{p}_x" for p in self.home_gk_players] if team_name.startswith("H") else [f"{p}_x" for p in self.away_gk_players]
        first_gk_x_columns = [p for p in first_row.index if p in gk_x_columns]
        
        if len(first_gk_x_columns) == 1:
            first_gk_x_column = first_gk_x_columns[0]
        else:
            raise ValueError(f"Expected exactly one goalkeeper for team {team_name}, but found {len(first_gk_x_columns)}: {first_gk_x_columns}")
        
        # +ve is left->right, -ve is right->left
        print(f"First row goalkeeper x position for team {team_name}, gk {first_gk_x_column}: {first_row[first_gk_x_column]}")
        return -np.sign(first_row[first_gk_x_column] - 52.5)
    
    def calc_ball_features(self, ball_traces: pd.DataFrame, use_default=False,
                           remove_outliers=True, smoothing=True) -> pd.DataFrame:
        
        # W_LEN = 11
        # P_ORDER = 2
        LOC_W_LEN = 9
        LOC_P_ORDER = 2
        SPEED_W_LEN = 9 #5
        SPEED_P_ORDER = 2
        ACCEL_W_LEN = 9 #9
        ACCEL_P_ORDER = 2
        MAX_BALL_SPEED = 28.0
        MAX_BALL_ACCELERATION = 13.5
        
        cols = ["frame", "time", "x", "y", "z", "vx", "vy", "vz", "speed", "accel"]
        
        if use_default:
            cols = ["frame", "time", "x", "y", "z", "speed", "accel"]
            x = ball_traces[self.BALL_x].values
            y = ball_traces[self.BALL_y].values
            z = ball_traces[self.BALL_z].values
            #vx = ball_traces[f"{self.BALL_x}_vx"].values
            #vy = ball_traces[f"{self.BALL_y}_vy"].values
            #vz = ball_traces[f"{self.BALL_z}_vz"].values
            speed = ball_traces[f"{self.BALL_x.split('_')[0]}_speed"].values
            accel = ball_traces[f"{self.BALL_x.split('_')[0]}_accel"].values
            
            ball_traces_arr = np.stack((ball_traces["frame_id"].values, ball_traces["time"].values, x, y, z, speed, accel), axis=1)
            return pd.DataFrame(ball_traces_arr, index=ball_traces.index, columns=cols)
        
        ball_traces = ball_traces.dropna(subset=[self.BALL_x])
        frames = ball_traces["frame_id"].values
        times = ball_traces["time"].values

        x = ball_traces[self.BALL_x].values
        y = ball_traces[self.BALL_y].values
        z = ball_traces[self.BALL_z].values
        if smoothing:
            if len(x) < LOC_W_LEN:
                LOC_W_LEN = len(x) if len(x) % 2 == 1 else len(x) - 1
                LOC_W_LEN = max(LOC_W_LEN, LOC_P_ORDER + 1)
                if LOC_W_LEN % 2 == 0:
                    LOC_W_LEN += 1
                SPEED_W_LEN = LOC_W_LEN
                ACCEL_W_LEN = LOC_W_LEN
                print(f"Adjusted LOC_W_LEN to {LOC_W_LEN} due to short length of x ({len(x)})")
            if len(x) <= LOC_P_ORDER:
                smoothing = False
                print(f"Skipping smoothing: len(x)={len(x)} <= polyorder={LOC_P_ORDER}")
            x = signal.savgol_filter(x, window_length=LOC_W_LEN, polyorder=LOC_P_ORDER)
            y = signal.savgol_filter(y, window_length=LOC_W_LEN, polyorder=LOC_P_ORDER)
            z = signal.savgol_filter(z, window_length=LOC_W_LEN, polyorder=LOC_P_ORDER)

        vx = np.diff(x, prepend=x[0]) / (1 / self.fps)
        vy = np.diff(y, prepend=y[0]) / (1 / self.fps)
        vz = np.diff(z, prepend=z[0]) / (1 / self.fps)
        
        if remove_outliers:
            speeds = np.sqrt(vx**2 + vy**2)
            is_speed_outlier = speeds > MAX_BALL_SPEED
            vx = pd.Series(np.where(is_speed_outlier, np.nan, vx)).interpolate(limit_direction="both").values
            vy = pd.Series(np.where(is_speed_outlier, np.nan, vy)).interpolate(limit_direction="both").values
            vz = pd.Series(np.where(is_speed_outlier, np.nan, vz)).interpolate(limit_direction="both").values

        if smoothing:
            vx = signal.savgol_filter(vx, window_length=SPEED_W_LEN, polyorder=SPEED_P_ORDER)
            vy = signal.savgol_filter(vy, window_length=SPEED_W_LEN, polyorder=SPEED_P_ORDER)
            vz = signal.savgol_filter(vz, window_length=SPEED_W_LEN, polyorder=SPEED_P_ORDER)
            
        speeds = np.sqrt(vx**2 + vy**2 + vz**2)

        accels = np.diff(speeds, prepend=speeds[-1]) / (1 / self.fps)
        accels[:2] = 0
        accels[-2:] = 0
        if smoothing:
            accels = signal.savgol_filter(accels, window_length=ACCEL_W_LEN, polyorder=ACCEL_P_ORDER)
            
        
        ball_traces_arr = np.stack((frames, times, x, y, z, vx, vy, vz, speeds, accels), axis=1)

        return pd.DataFrame(ball_traces_arr, index=ball_traces.index, columns=cols)
    
    def calc_ball_dists(self, traces: pd.DataFrame, players: list) -> pd.DataFrame:
        # Calculate distances from the ball to the players
        player_xy_cols = [f"{p}{t}" for p in players for t in ["_x", "_y"]]
        player_xy = traces[player_xy_cols].values.reshape(traces.shape[0], -1, 2)
        pred_xy = traces[[self.BALL_x, self.BALL_y]].values[:, np.newaxis, :]
        ball_dists = np.linalg.norm(pred_xy - player_xy, axis=-1)
        ball_dists = pd.DataFrame(ball_dists, index=traces.index, columns=players)

        # Calculate distances from the ball to the pitch lines
        ball_dists["OUT-L"] = (traces["OUT-L_x"] - traces[self.BALL_x]).abs()
        ball_dists["OUT-R"] = (traces["OUT-R_x"] - traces[self.BALL_x]).abs()
        ball_dists["OUT-B"] = (traces["OUT-B_y"] - traces[self.BALL_y]).abs()
        ball_dists["OUT-T"] = (traces["OUT-T_y"] - traces[self.BALL_y]).abs()

        return ball_dists

    def generate_touch_records(self, records: pd.DataFrame) -> pd.DataFrame:
        """
            control_records: 공이 급격하게 감속 또는 가속하는 시점(control_idx)을 활용하여 receive / interception 이벤트를 검출
                - receive: control_idx 시점에서 player가 바뀌는 경우, 해당 시점에 receive 이벤트가 발생했다고 가정한다. (공을 받는 선수는 이전 player와 다른 팀원 선수)
                - interception: control_idx 시점에서 player가 바뀌는 경우, 해당 시점에 interception 이벤트가 발생했다고 가정한다. (공을 받는 선수는 이전 player와 다른 상대팀의 선수)
        """
        
        if records.empty:
            return pd.DataFrame(columns=self.event_records_cols)
        
        poss_change = (
            (records.subtype == "control") & # convert control action to receive or interception event when player changes. 
            (records.player != records.player.shift(1)) # shift(1): previous player. only detect first touch when player changes.
        )

        touch_rows = records[poss_change].copy()
        prev_touch_rows = records[poss_change.shift(-1, fill_value=False)].copy() # to get previous player info, shift(-1): next player.

        if touch_rows.empty:
            return pd.DataFrame(columns=self.event_records_cols)
        
        control_records = pd.DataFrame({
            "start_idx": touch_rows["control_idx"].values,
            "end_idx": touch_rows["control_idx"].values,
            "player": touch_rows["player"].values,
            "receiver": None,
            "type": "control",
            "subtype": np.where(
                touch_rows["player"].str[0].values == prev_touch_rows["player"].str[0].values,
                "receive",
                "interception"
            ),
            "success": True
        })
        
        return control_records
        

    def generate_carry_records(self, records: pd.DataFrame) -> pd.DataFrame:
        """
            control_records: 공이 급격하게 감속 또는 가속하는 시점(control_idx)과 해당 시점에서의 carrier 후보들(candidate_carriers)을 담은 데이터프레임
            control_records에가 연속된 두 player 가 검출되면 해당 구간을 carry로 정의하여 carry_records에 저장하는 방식으로 carry_records 생성
            서로 다른 두 player가 검출되면 carry 정보 없음으로 간주하여 carry_records에 저장하지 않음
        """

        if records.empty:
            return pd.DataFrame(columns=self.event_records_cols)

        poss_not_change = (
            (records["player"] != records["player"].shift(1)) # shift(1): prev player. only consider carry when player does not change to another player.
        )

        # if continuous same player, assign same group_id. if player changes, assign different group_id.
        records['group_id'] = poss_not_change.cumsum()
        control_grouped = records.groupby('group_id').agg(
            start_idx=('control_idx', 'first'),    # first control_idx in the group
            end_idx=('control_idx', 'last'),       # last control_idx in the group
            player=('player', 'first'),
            control_count=('control_idx', 'count') # number of controls in the group
        ).reset_index(drop=True)

        carry_rows = control_grouped[control_grouped['control_count'] > 1].copy()
        if carry_rows.empty:
            return pd.DataFrame(columns=self.event_records_cols)

        carry_records = pd.DataFrame({
            "start_idx": carry_rows["start_idx"].values,
            "end_idx": carry_rows["end_idx"].values,
            "player": carry_rows["player"].values,
            "receiver": None,
            "type": "control",
            "subtype": "carry",
            "success": True
        })

        return carry_records
    
    def generate_kick_records(self, records: pd.DataFrame) -> pd.DataFrame:
        """
            control_records에서 연속된 player가 검출되는 구간을 carry로 정의하는 방식과 달리,
            control_records에서 player가 바뀌는 시점(control_idx)을 kick으로 정의하여 kick_records에 저장하는 방식으로 kick_records 생성
        """
        if records.empty:
            return pd.DataFrame(columns=self.event_records_cols)

        poss_change = (
            (~records.subtype.isin(["out", "pause"])) &
            (records.player != records.player.shift(-1)) # shift(-1): next player. only detect kick when player changes to another player. 
        )

        kick_rows = records[poss_change].copy()
        next_kick_rows = records[poss_change.shift(1, fill_value=False)].copy() # to get next player info, shift(1): previous player.
        
        if kick_rows.empty:
            return pd.DataFrame(columns=self.event_records_cols)

        kick_records = pd.DataFrame({
            "start_idx": kick_rows["control_idx"].values,
            "end_idx": next_kick_rows["control_idx"].values,
            "player": kick_rows["player"].values,
            "receiver": next_kick_rows["player"].values,
            "type": "kick", 
            "subtype": ["kick" if t == "control" else t for t in kick_rows["subtype"].values], # kick or set_piece (corner, throw_in, etc.)
            "success": np.where(
                kick_rows["player"].str[0].values == next_kick_rows["player"].str[0].values,
                True,
                False
            )
        })

        return kick_records
    
    def generate_out_records(self, control_records: pd.DataFrame) -> pd.DataFrame:
        """
            control_records에서 player가 pitch line(OUT-L, OUT-R, OUT-B, OUT-T)으로 바뀌는 시점(control_idx)을 out으로 정의하여 out_records에 저장하는 방식으로 out_records 생성
        """
    
        if control_records is None or control_records.empty:
            return pd.DataFrame(columns=self.event_records_cols)
        
        players = control_records["player"].copy()
        out_change = players.str.startswith("OUT")
        pause_change = players.str.startswith("PAUSE")

        out_rows = control_records.loc[out_change | pause_change].copy()
        if out_rows.empty:
            return pd.DataFrame(columns=self.event_records_cols)
        
        out_records = pd.DataFrame({
            "start_idx": out_rows["control_idx"].values,
            "end_idx": out_rows["control_idx"].values,
            "player": out_rows["player"].values,
            "receiver": None,
            "type": "out",
            "subtype": out_rows["subtype"].values, # out or pause
            "success": None
        })

        return out_records

    def generate_shot_records(self, traces: pd.DataFrame, kick_records: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
        """
            pass, shot, cross, save, punch 이벤트 검출을 위한 알고리즘
            1) type = kick 중, receiver = "OUT-L", "OUT-R", "control (골키퍼)"이면서, pass_zone(cross_zone)이 아닌 경우 → subtype = shot
            2) type = kick & subtype != shot 중, cross zone에 위치한 경우 → subtype = cross, 나머지 → subtype = pass
         
        """
        # Pitch Boundary Coordinates (0,0) is at the bottom-left corner of the pitch, and (105, 68) is at the top-right corner of the pitch.
        pitch_left_x, center_x, pitch_right_x = 0, self.pitch_size[0] / 2, self.pitch_size[0]
        pitch_bottom_y, center_y, pitch_top_y = 0, self.pitch_size[1] / 2, self.pitch_size[1]
        
        field_half_dx = self.pitch_size[0] / 2
        field_half_w = self.pitch_size[1] / 2
        center_circle_radius = 9.14
        penalty_area_half_w = 20.16
        zone_tol = 1.0
        zone_spot_dx = field_half_dx - center_circle_radius
        zone_spot_w = field_half_w - penalty_area_half_w
        
        goal_width = 7.32
        goal_length = 2.44
        goal_height = 2.44
        
        period_id = traces["period_id"].iloc[0]
        home_direction = self.direction_dict["Home"][period_id]
        away_direction = self.direction_dict["Away"][period_id]
                
        if kick_records.empty:
            #print("No kick records found. Returning empty DataFrames for pass, cross, shot, and keeper records.")
            return pd.DataFrame(columns=self.event_records_cols)
        
        home_gk_players = [p for p in players if p in self.home_gk_players]
        away_gk_players = [p for p in players if p in self.away_gk_players]

        attack_zones = {
            "left": {
                "shot_area": Polygon([
                    (pitch_left_x - zone_tol, pitch_bottom_y + zone_spot_w),
                    (pitch_left_x + zone_spot_dx + zone_tol, pitch_bottom_y + zone_spot_w),
                    (pitch_left_x + zone_spot_dx + zone_tol, pitch_top_y - zone_spot_w),
                    (pitch_left_x - zone_tol, pitch_top_y - zone_spot_w)
                ]), # left shot area
                "cross_areas": [
                    Polygon([
                        (pitch_left_x - zone_tol, pitch_bottom_y - zone_tol),
                        (pitch_left_x + zone_spot_dx + zone_tol, pitch_bottom_y - zone_tol),
                        (pitch_left_x + zone_spot_dx + zone_tol, pitch_bottom_y + zone_spot_w),
                        (pitch_left_x - zone_tol, pitch_bottom_y + zone_spot_w)
                    ]), # left bottom corner area
                    Polygon([
                        (pitch_left_x, pitch_top_y - zone_spot_w),
                        (pitch_left_x + zone_spot_dx + zone_tol, pitch_top_y - zone_spot_w),
                        (pitch_left_x + zone_spot_dx + zone_tol, pitch_top_y + zone_tol),
                        (pitch_left_x - zone_tol, pitch_top_y + zone_tol)
                    ]) # left top corner area
                ], # left cross areas
                "goal_post_area": Polygon([
                    (pitch_left_x - goal_length - zone_tol, center_y - goal_width / 2 - zone_tol),
                    (pitch_left_x + zone_tol, center_y - goal_width / 2 - zone_tol),
                    (pitch_left_x + zone_tol, center_y + goal_width / 2 + zone_tol),
                    (pitch_left_x - goal_length - zone_tol, center_y + goal_width / 2 + zone_tol)
                ]) # left goal post area
            },
            "right": {
                "shot_area": Polygon([
                    (pitch_right_x - zone_spot_dx - zone_tol, pitch_bottom_y + zone_spot_w),
                    (pitch_right_x + zone_tol, pitch_bottom_y + zone_spot_w),
                    (pitch_right_x + zone_tol, pitch_top_y - zone_spot_w),
                    (pitch_right_x - zone_spot_dx - zone_tol, pitch_top_y - zone_spot_w)
                ]), # right shot area
                "cross_areas": [
                    Polygon([
                        (pitch_right_x - zone_spot_dx - zone_tol, pitch_bottom_y - zone_tol),
                        (pitch_right_x + zone_tol, pitch_bottom_y - zone_tol),
                        (pitch_right_x + zone_tol, pitch_bottom_y + zone_spot_w),
                        (pitch_right_x - zone_spot_dx - zone_tol, pitch_bottom_y + zone_spot_w)
                    ]), # right bottom corner area
                    Polygon([
                        (pitch_right_x - zone_spot_dx - zone_tol, pitch_top_y - zone_spot_w),
                        (pitch_right_x + zone_tol, pitch_top_y - zone_spot_w),
                        (pitch_right_x + zone_tol, pitch_top_y + zone_tol),
                        (pitch_right_x - zone_spot_dx - zone_tol, pitch_top_y + zone_tol)
                    ]) # right top corner area
                ], # right cross areas
                "goal_post_area": Polygon([
                    (pitch_right_x - zone_tol, center_y - goal_width / 2 - zone_tol),
                    (pitch_right_x + goal_length + zone_tol, center_y - goal_width / 2 - zone_tol),
                    (pitch_right_x + goal_length + zone_tol, center_y + goal_width / 2 + zone_tol),
                    (pitch_right_x - zone_tol, center_y + goal_width / 2 + zone_tol)
                ]) # right goal post area
            }
        }

        for idx, record in kick_records.iterrows():
            player = record.player
            receiver = record.receiver
            subtype = record.subtype    
            
            if player is None:
                kick_records.at[idx, "subtype"] = None
                continue
            
            defending_gk_player = away_gk_players if player.startswith("H") else home_gk_players
            attacking_direction = home_direction if player.startswith("H") else away_direction
            zone = attack_zones["right"] if attacking_direction > 0 else attack_zones["left"]
            shot_area, cross_areas, goal_post_area = zone["shot_area"], zone["cross_areas"], zone["goal_post_area"]
            
            start_ball_point = Point(traces.loc[record.start_idx, [self.BALL_x, self.BALL_y]])
            end_ball_point = Point(traces.loc[record.end_idx, [self.BALL_x, self.BALL_y]])
            ball_height = traces.loc[record.end_idx, self.BALL_z]

            if receiver in ["OUT-L", "OUT-R"] + defending_gk_player:
                if shot_area.contains(start_ball_point):
                    kick_records.at[idx, "subtype"] = "shot"
                    
                    # 골대 안으로 향하는 shot인 경우, 성공 여부 판단
                    # Shapely Library의 contains() 메서드는 3D 포인트에 대해서는 z 좌표를 고려하지 않고, 2D 평면에서의 포함 여부만을 판단한다. 이에 높이 정보를 별도로 고려해야한다.
                    if (
                        (goal_post_area.contains(end_ball_point)) and 
                        (ball_height <= goal_height + zone_tol)
                    ):
                        kick_records.at[idx, "success"] = True
                    else:
                        kick_records.at[idx, "success"] = False
                elif any(cross_area.contains(start_ball_point) for cross_area in cross_areas):
                    kick_records.at[idx, "subtype"] = "cross" if subtype == "kick" else subtype
                else:
                    kick_records.at[idx, "subtype"] = "pass" if subtype == "kick" else subtype
            else:
                kick_records.at[idx, "subtype"] = "pass" if subtype == "kick" else subtype
   
        return kick_records    

    def generate_keeper_records(self, traces: pd.DataFrame, touch_records: pd.DataFrame, kick_records: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
        """
            save, punch 이벤트 검출을 위한 알고리즘
            1) type = kick 중, receiver = "control (골키퍼)"인 경우 → subtype = save or punch
            2) control 이후 kick 액션까지 수행한 경우, save, control 액션만 수행한 경우 punch로 정의
        """
        period_id = traces["period_id"].iloc[0]
        home_direction = self.direction_dict["Home"][period_id]
        away_direction = self.direction_dict["Away"][period_id]
        
        # Pitch Boundary Coordinates (0,0) is at the bottom-left corner of the pitch, and (105, 68) is at the top-right corner of the pitch.
        pitch_left_x, center_x, pitch_right_x = 0, self.pitch_size[0] / 2, self.pitch_size[0]
        pitch_bottom_y, center_y, pitch_top_y = 0, self.pitch_size[1] / 2, self.pitch_size[1]
        
        if kick_records.empty:
            #print("No kick records found. Returning empty DataFrame for keeper records.")
            return pd.DataFrame(columns=self.event_records_cols)
        
        home_gk_players = [p for p in players if p in self.home_gk_players]
        away_gk_players = [p for p in players if p in self.away_gk_players]

        # Penalty Area tolerance
        penalty_area_half_w = 20.16
        penalty_area_dx = 16.5
        penalty_area_tol = 2.0
        
        penalty_area_left = Polygon([
            (pitch_left_x - penalty_area_tol, center_y - penalty_area_half_w - penalty_area_tol), 
            (pitch_left_x + penalty_area_dx + penalty_area_tol, center_y - penalty_area_half_w - penalty_area_tol), 
            (pitch_left_x + penalty_area_dx + penalty_area_tol, center_y + penalty_area_half_w + penalty_area_tol), 
            (pitch_left_x - penalty_area_tol, center_y + penalty_area_half_w + penalty_area_tol)        
        ])
        penalty_area_right = Polygon([
            (pitch_right_x - penalty_area_dx - penalty_area_tol, center_y - penalty_area_half_w - penalty_area_tol), 
            (pitch_right_x + penalty_area_tol, center_y - penalty_area_half_w - penalty_area_tol), 
            (pitch_right_x + penalty_area_tol, center_y + penalty_area_half_w + penalty_area_tol), 
            (pitch_right_x - penalty_area_dx - penalty_area_tol, center_y + penalty_area_half_w + penalty_area_tol)
        ])  
        
        for idx, record in kick_records.iterrows():
            player = record.player
            receiver = record.receiver
            subtype = record.subtype
            
            if player is None:
                continue
            
            defending_gk_player = away_gk_players if player.startswith("H") else home_gk_players
            defending_gk_player_points = [Point(traces.loc[record.end_idx, [f"{gk}_x", f"{gk}_y"]]) for gk in defending_gk_player]
            
            attacking_direction = home_direction if player.startswith("H") else away_direction
            penalty_area = penalty_area_right if attacking_direction > 0 else penalty_area_left
            
            if (
                (receiver in defending_gk_player) and 
                any(penalty_area.contains(gk_point) for gk_point in defending_gk_player_points) # receiver가 골키퍼이면서, 해당 receiver의 위치가 페널티 에어리어 안에 있는 경우
            ):
                # control 이후 kick 액션까지 수행한 경우 save, control 액션만 수행한 경우 punch로 정의
                subsequent_control = touch_records[
                    (touch_records["type"] == "control") &
                    (touch_records["player"] == receiver) &
                    (touch_records["start_idx"] == record["end_idx"]) # keeper'control frame == kick frame
                ]

                if subsequent_control.empty:
                    raise ValueError(f"Expected at least one subsequent control record for receiver {receiver} at index {idx}, but found none.")
                else:
                    subsequent_control = subsequent_control.iloc[0]
                    
                if subsequent_control.subtype == "receive":
                    continue # 키퍼가 receive한 경우는 save/punch로 간주하지 않고, pass/cross/shot의 수신으로 간주하여 keeper_records에 저장하지 않음
                
                if subsequent_control.subtype == "interception":
                    touch_records.at[subsequent_control.name, "subtype"] = "save" 
                    touch_records.at[subsequent_control.name, "success"] = True
                
        return touch_records
        
    def detect_control_by_accel(
            self, ball_feats: pd.DataFrame, ball_dists: pd.DataFrame = None, max_accel=5
        ) -> pd.DataFrame:
        """
            본 알고리즘은 이벤트 라벨 없이 트래킹 데이터만을 이용하여 carry 이벤트를 휴리스틱하게 검출하는 방법이다.

            핵심 가정은 하나의 carry가 공의 운동 상태 변화, 
            즉 가속도의 유의미한 감속(local minimum) 이후 가속(local maximum)까지의 연속 구간으로 나타난다는 것이다.
            따라서 carry는 시간적으로 가장 가까운 (min → max) 가속도 극값 쌍으로 정의된다.

            예를 들어, 가속도 극값 시퀀스가 min(1) → max(4) → min(5) → max(10) 인 경우,
            [min(1), max(4)]를 carry1, [min(5), max(10)]를 carry2로 정의한다.

            i) 하나의 carry 구간 내에서 감속(min) 또는 가속(max)이 시간적으로 연속해서 여러 번 검출될 수 있으므로,
                - 현재 max 이전에 발생한 모든 min을 하나의 min_group으로 묶고, 그 중 가장 작은 가속도 값을 갖는 시점(idxmin)을 carry 시작점으로 선택한다.
                - 현재 min 이전에 발생한 모든 max를 하나의 max_group으로 묶고, 그 중 가장 큰 가속도 값을 갖는 시점(idxmax)을 carry 종료점으로 선택한다.

            ii) local maximun이 처음에 등장하거나 local minimum이 마지막에 등장하는 경우, 시작 or 끝 가속도 값으로 보간한다.
                - 첫 극값이 max인 경우, carry 시작이 이전에 존재했다고 가정하고 ball_feats.index[0]을 해당 carry의 시작(min)으로 사용한다.
                - 마지막 극값이 min인 경우, carry 종료가 이후에 존재했다고 가정하고 ball_feats.index[-1]을 해당 carry의 종료(max)로 사용한다.
        """

        player_dist_func = Postprocessor.linear_scoring_func(0, 3, increasing=False)
        player_height_func = Postprocessor.linear_scoring_func(0, 1.5, increasing=False)

        accels = ball_feats[["accel"]].copy()

        # find_peaks parameter description:
        # - prominence: 봉우리의 돌출 정도를 나타내는 값으로, 봉우리의 높이와 주변의 골짜기 깊이의 차이를 기준으로 봉우리를 검출하는 데 사용됩니다. 값이 클수록 더 뚜렷한 봉우리를 검출합니다.
        # - width: 봉우리의 너비를 기준으로 봉우리를 검출하는 데 사용됩니다. 값이 클수록 더 넓은 봉우리를 검출합니다.
        # - rel_height: 봉우리의 상대 높이를 기준으로 봉우리를 검출하는 데 사용됩니다. 값이 클수록 봉우리의 높이가 주변보다 더 높아야 검출됩니다.
        # - distance: 봉우리 간의 최소 거리를 기준으로 봉우리를 검출하는 데 사용됩니다. 값이 클수록 봉우리 간의 간격이 넓어야 검출됩니다.
        # max_peaks, _ = find_peaks(ball_feats["accel"], prominence=3, width=1, rel_height=0.3, distance=10)
        # min_peaks, _ = find_peaks(-ball_feats["accel"], prominence=3, width=1, rel_height=0.3, distance=10)
        # max_peaks, _ = find_peaks(ball_feats["accel"], prominence=5, width=1, rel_height=0.5, distance=10)
        # min_peaks, _ = find_peaks(-ball_feats["accel"], prominence=5, width=1, rel_height=0.5, distance=10)
        max_peaks, _ = find_peaks(ball_feats["accel"], prominence=5, width=1, rel_height=0.5, distance=10)
        min_peaks, _ = find_peaks(-ball_feats["accel"], prominence=5, width=1, rel_height=0.5, distance=10)
        max_idxs = accels.iloc[max_peaks].index.tolist()
        min_idxs = accels.iloc[min_peaks].index.tolist()

        peak_idx = list(sorted(max_idxs + min_idxs))

        # 첫번째 프레임이 세트피스로 인해 local maximum으로 검출되는 경우, 해당 극값은 실제 control 시작점이 아니라고 판단하여 제거한다. (control 시작점은 local minimum이어야 함)
        if peak_idx and peak_idx[0] in max_idxs:
            peak_idx = peak_idx[1:]


        # 첫번째 local minimum 이전의 세트피스 상황이 실제로 검출이 되었는지
        # set_piece_speed_threshold = 5m/s로 설장하여 첫번째 local minimum 이전에 공의 속도가 set_piece_speed_threshold 이상으로 올라가야함
        # set_piece 상황이 명확하게 검출되지 않고, 선수가 공을 컨트롤하는 상황이 local maximum으로 검출되는 경우가 존재할 수 있기 때문에, 첫번째 local minimum 이전의 set_piece_speed_threshold 이상으로 올라가지 않는 경우 해당 local minimum을 제거한다.
        # 첫번째 local minimum 이전에 공의 속도가 set_piece_speed_threshold 이상으로 올라가지 않는 경우, 해당 local minimum이 실제 control 시작점이 아닐 가능성이 높다고 판단하여 제거한다.
        # 예를 들어, [2, 5, 10, 15] 프레임에서 local minimum이 검출되고, 2, 5프레임 이전에 공의 속도가 set_piece_speed_threshold 이상으로 올라가지 않는 경우, 해당 local minimum인 2, 5프레임이 실제 control 시작점이 아닐 가능성이 높다고 판단하여 제거한다. 즉, 10 프레임 부터 control 시작점으로 판단한다.
        # while 과정에서 local maximum이 나타나는 경우, 제거
        set_piece_speed_threshold = 5.0
        while peak_idx:
            first_peak_idx = peak_idx[0]
            if ball_feats.loc[:first_peak_idx, "speed"].max() < set_piece_speed_threshold:
                peak_idx = peak_idx[1:]
            elif first_peak_idx in max_idxs:
                peak_idx = peak_idx[1:]
            else:
                break
            
        if not peak_idx:
            return pd.DataFrame(columns=self.control_records_cols)

        control_records = []
        for i in range(len(peak_idx)):
            control_idx = peak_idx[i]
            
            # 두 점수의 합이 가장 높은 선수을 carrier로 선정
            use_cols = [c for c in ball_dists.columns if c not in ["OUT-L", "OUT-R", "OUT-B", "OUT-T"]]
            window_index = ball_feats[ball_feats["z"] < 3.5].loc[control_idx-2:control_idx+2].index

            player_dist_score = player_dist_func(ball_dists.loc[window_index, use_cols]).max()
            total_scores = player_dist_score

            control_score = total_scores.max()
            player = total_scores.idxmax()

            if pd.isna(control_score) or control_score < 0.5:
                continue
  
            control_records.append({
                "control_idx": control_idx,
                "player": player,
                "type": "control",
                "subtype": "control",
                "control_score": np.round(control_score, 3),
                #"player_dist_score": np.round(player_dist_score[player], 3),
                "candidate_players": ",".join(total_scores.sort_values(ascending=False).index[1:4].tolist()),
                #"candidate_control_scores": ",".join(total_scores.sort_values(ascending=False).iloc[1:4].round(3).astype(str).tolist()),
                #"candidate_player_dist_scores": ",".join(player_dist_score.sort_values(ascending=False).iloc[1:4].round(3).astype(str).tolist()),
            })

        return pd.DataFrame(control_records)

    def detect_set_piece_by_distance(
            self, traces: pd.DataFrame, ball_feats: pd.DataFrame, control_records: pd.DataFrame, players: list
        ) -> pd.DataFrame:
        """
            본 알고리즘은 이벤트 라벨 없이 트래킹 데이터만을 이용하여 set piece 이벤트를 휴리스틱하게 검출하는 방법이다.
            Corner Kick, Goal Kick, Throw In과 같이 공이 경기장 구역 특정 위치에서 시작되는 set piece 상황은 공의 초기 위치를 이용하여 검출할 수 있다고 가정한다.
            6가지의 set piece 상황을 검출하기 위한 휴리스틱 트리거는 다음과 같이 정의된다. (IFAB Laws of the Game 참조)
            각 세트피스가 발생할 수 있는 최소한의 공간적인 조건을 만족하는 경우, 해당 세트피스 상황이 발생했다고 가정한다.

            참고 문헌
                - kickoff trigger: all players are within their own halves (with a tolerance of k1) and 
                there is at least one player within k2 of the center mark, according to IFAB Law 8.

                – penalty kick trigger: only one player is at their goal line between the posts (with tolerance bounding box of p1),
                only one opponent is within a square bounding box from p2∕4 in front to 3p2∕4 behind the active penalty mark,
                the other players are neither within the penalty area nor within 9.15 m from the penalty mark (with a tolerance of p3), according to IFAB Law 14
                
                – goal kick trigger: at least one player is within their own goal area (with tolerance bounding box of c), 
                according to IFAB Law 16..
                
                – corner kick trigger: at least one player is within c of one of their active corner marks, according to IFAB Law 17.
                
                – throw-in trigger: at least one player is beyond the auxiliary sideline (sideline minus t), according to IFAB Law 15.
        """
        period_id = traces["period_id"].iloc[0]
        home_direction = self.direction_dict["Home"][period_id]
        away_direction = self.direction_dict["Away"][period_id]
        
        if control_records.empty:
            #print(f"No control records detected. Skipping set piece detection. Frame: {ball_feats.index[0]}~{ball_feats.index[-1]}")
            return pd.DataFrame(columns=self.control_records_cols)
        
        first_control_idx = control_records.iloc[0]["control_idx"]
        set_piece_idx = max(ball_feats.index[0], first_control_idx - 1)
        for idx in range(first_control_idx - 1, ball_feats.index[0] - 1, -1):
            speed = ball_feats.at[idx, "speed"]
            
            if speed < 0.5: # Beginning of set piece is likely to be a moment when the ball is stationary or nearly stationary.
                set_piece_idx = idx
                break
        
        # No significant deceleration found before first control, use first frame as set piece start
        if set_piece_idx == first_control_idx - 1:
            set_piece_idx = ball_feats.index[0]
            
        # Pitch Boundary Coordinates (0,0) is at the bottom-left corner of the pitch, and (105, 68) is at the top-right corner of the pitch.
        pitch_left_x, center_x, pitch_right_x = 0, self.pitch_size[0] / 2, self.pitch_size[0]
        pitch_bottom_y, center_y, pitch_top_y = 0, self.pitch_size[1] / 2, self.pitch_size[1]

        # Player and Ball Coordinates at the set piece moment
        ball_point = Point(traces.loc[set_piece_idx, [self.BALL_x, self.BALL_y]])
        player_points = {pid: Point(traces.loc[set_piece_idx, f"{pid}_x"], traces.loc[set_piece_idx, f"{pid}_y"]) for pid in players}
        home_player_points = {pid: p for pid, p in player_points.items() if pid.startswith("H")}    
        away_player_points = {pid: p for pid, p in player_points.items() if pid.startswith("A")}
        
        # GoalKeeper Player Points: exist one goalkeeper for each team, but in case of multiple goalkeepers detected.
        home_gk_player_points = {pid: p for pid, p in home_player_points.items() if pid in self.home_gk_players}    
        away_gk_player_points = {pid: p for pid, p in away_player_points.items() if pid in self.away_gk_players}
        if len(home_gk_player_points) == 1:
            home_gk_pid, home_gk_player_point = list(home_gk_player_points.items())[0]
        else:     
            home_gk_pid, home_gk_player_point = list(home_gk_player_points.items())[0]
            #print(f"Expected exactly one home goalkeeper, but found {len(home_gk_player_points)}: {home_gk_player_points}. Frame: {ball_feats.index[0]}~{ball_feats.index[-1]}, Set Piece Index: {set_piece_idx}")
          
        if len(away_gk_player_points) == 1:
            away_gk_pid, away_gk_player_point = list(away_gk_player_points.items())[0]
        else:
            away_gk_pid, away_gk_player_point = list(away_gk_player_points.items())[0]
            #print(f"Expected exactly one away goalkeeper, but found {len(away_gk_player_points)} players: {away_gk_player_points}. Frame: {ball_feats.index[0]}~{ball_feats.index[-1]}, Set Piece Index: {set_piece_idx}")
           
        def build_candidates(candidate_ids: list[str], score_scale: float = 0.0) -> tuple[str, float, str]:
            """
                candidate_ids: set piece type에 따라 후보 선수들의 ID 리스트
                score_scale: 후보 선수들의 점수 계산 시, 공과의 최대 유의미한 거리로 사용한다. 
            """
            if len(candidate_ids) == 0:
                #print(f"No candidates found for set piece type. Frame: {ball_feats.index[0]}~{ball_feats.index[-1]}, Candidate IDs: {candidate_ids}, Score Scale: {score_scale}")
                return None, 0.0, None
                
            candidate_ids = sorted(candidate_ids, key=lambda pid: player_points[pid].distance(ball_point))
            player = candidate_ids[0]
            control_score = max(0.0, 1.0 - player_points[player].distance(ball_point) / max(score_scale, 1e-6)) # 0 ~ 1 사이의 점수로 변환. score_scale 거리 이내에 있을수록 1에 가까운 점수를 받는다.
            candidate_players = ",".join(candidate_ids[1:4]) 
            return player, control_score, candidate_players

        # 1) Corner Kick: 공이 코너 플래그 근처에 있고, 해당 코너에 공격팀 선수 중 최소 한 명이 존재하는 경우, 해당 코너에 가장 가까운 선수 중에서 공과의 거리가 가장 가까운 선수를 corner kick을 수행한 선수로 간주한다.
        corner_mark_tol = 3   
        corner_areas = {
            "bottom-left": Point(pitch_left_x, pitch_bottom_y).buffer(corner_mark_tol),  # bottom-left corner circle
            "top-left": Point(pitch_left_x, pitch_top_y).buffer(corner_mark_tol),     # top-left corner circle
            "bottom-right": Point(pitch_right_x, pitch_bottom_y).buffer(corner_mark_tol), # bottom-right corner circle
            "top-right": Point(pitch_right_x, pitch_top_y).buffer(corner_mark_tol)     # top-right corner circle
        }
        for corner_name, corner_area in corner_areas.items():
            if corner_name in ["bottom-left", "top-left"]:
                attacking_player_points = away_player_points if home_direction > 0 else home_player_points # home team is left -> right.
            elif corner_name in ["bottom-right", "top-right"]:
                attacking_player_points = home_player_points if home_direction > 0 else away_player_points # home team is left <- right.
            else:
                raise ValueError(f"Unexpected corner name: {corner_name}. Frame: {ball_feats.index[0]}~{ball_feats.index[-1]}, Set Piece Index: {set_piece_idx}")
        
            if (
                corner_area.contains(ball_point) and
                any(corner_area.contains(p) for p in attacking_player_points.values()) # 해당 코너에 공격팀 선수 중 최소 한 명이 존재하는 경우
            ):
     
                corner_candidates = [pid for pid, p in attacking_player_points.items() if corner_area.contains(p)]
                player, control_score, candidate_players = build_candidates(corner_candidates, score_scale=corner_mark_tol) 
  
                return pd.DataFrame([
                    {
                        "control_idx": set_piece_idx,
                        "player": player,
                        "type": "control",
                        "subtype": "corner_kick",
                        "control_score": np.round(control_score, 3),
                        "candidate_players": candidate_players,
                    }
                ])
            
        # 2) Throw In: 공이 터치라인 (사이드 라인) 근처에 있으면, 해당 사이드 라인에 가장 가까운 선수 중에서 공과의 거리가 가장 가까운 선수를 throw-in을 수행한 선수로 간주한다.
        sideline_tol = 1.5
        sideline_areas = [
            Polygon([
                (pitch_left_x, pitch_bottom_y - sideline_tol), (pitch_right_x, pitch_bottom_y - sideline_tol), 
                (pitch_right_x, pitch_bottom_y + sideline_tol), (pitch_left_x, pitch_bottom_y + sideline_tol)
            ]), # bottom sideline
            Polygon([
                (pitch_left_x, pitch_top_y - sideline_tol), (pitch_right_x, pitch_top_y - sideline_tol), 
                (pitch_right_x, pitch_top_y + sideline_tol), (pitch_left_x, pitch_top_y + sideline_tol)
            ])  # top sideline
        ]
        for sideline_area in sideline_areas:
            if sideline_area.contains(ball_point):
                # Throw-In은 공간적인 조건만으로는 공격팀과 수비팀이 명확하게 구분되지 않으므로, 양 팀 선수 모두를 후보로 고려한다.
                sideline_candidates = [pid for pid, p in player_points.items() if sideline_area.contains(p)]
                player, control_score, candidate_players = build_candidates(sideline_candidates, score_scale=sideline_tol) 

                return pd.DataFrame([
                    {
                        "control_idx": set_piece_idx,
                        "player": player,
                        "type": "control",
                        "subtype": "throw_in",
                        "control_score": np.round(control_score, 3),
                        "candidate_players": candidate_players,
                    }
                ])

        # 3) Kick Off: 공이 센터 영역 근처에 있고, 오직 한 명의 선수가 센터 영역 안에 있으며, 양 팀 선수들이 각각 자신의 진영에 위치해 있는 경우, 센터 마크 근처에 위치한 선수를 kick-off를 수행한 선수로 간주한다.
        center_mark_tol = 2.0
        own_half_tol = 3.0
        center_area = Point(center_x, center_y).buffer(center_mark_tol)

        left_half_area = Polygon([
            (pitch_left_x - own_half_tol, pitch_bottom_y - own_half_tol), (center_x + own_half_tol, pitch_bottom_y - own_half_tol), 
            (center_x + own_half_tol, pitch_top_y + own_half_tol), (pitch_left_x - own_half_tol, pitch_top_y + own_half_tol)
        ]) # left half area
        right_half_area = Polygon([
            (center_x - own_half_tol, pitch_bottom_y - own_half_tol), (pitch_right_x + own_half_tol, pitch_bottom_y - own_half_tol), 
            (pitch_right_x + own_half_tol, pitch_top_y + own_half_tol), (center_x - own_half_tol, pitch_top_y + own_half_tol)
        ]) # right half area

        # print(center_area.contains(ball_point))
        # for pid, p in home_player_points.items():
        #     print(f"Home Player {pid} at {p}, distance to ball: {left_half_area.contains(p)}, {right_half_area.contains(p)}")
        
        # for pid, p in away_player_points.items():
        #     print(f"Away Player {pid} at {p}, distance to ball: {left_half_area.contains(p)}, {right_half_area.contains(p)}")
            
        # dd
            
            
        if (
            center_area.contains(ball_point) and
            (sum(center_area.contains(p) for p in player_points.values()) == 1) and
            (
                (
                    all(left_half_area.contains(p) for p in home_player_points.values()) and 
                    all(right_half_area.contains(p) for p in away_player_points.values())
                ) or
                (
                    all(right_half_area.contains(p) for p in home_player_points.values()) and 
                    all(left_half_area.contains(p) for p in away_player_points.values())
                )
            )
        ):
            center_candidates = [pid for pid, p in player_points.items() if center_area.contains(p)]
            player, control_score, candidate_players = build_candidates(center_candidates, score_scale=center_mark_tol) 

            return pd.DataFrame([
                {
                    "control_idx": set_piece_idx,
                    "player": player,
                    "type": "control",
                    "subtype": "kick_off",
                    "control_score": np.round(control_score, 3),
                    "candidate_players": candidate_players,
                }
            ])
            
        # 4) Penalty Kick: 공이 페널티 마크 근처에 있고, 오직 한 명의 공격팀 선수가 페널티 에어리어 안에 있으며, 수비팀 골키퍼가 골 에어리어 안에 있는 경우, 페널티 마크 근처에 위치한 선수를 penalty kick을 수행한 선수로 간주한다.
        penalty_spot_dx = 11.0
        penalty_area_dx = 16.5
        goal_area_dx = 5.5
        penalty_area_half_w = 20.16
        goal_area_half_w = 9.16

        # Penalty / goal-kick 판별용 tolerance
        goal_area_tol = 0.5
        penalty_mark_tol = 2.0
        penalty_area_tol = 2.0
        
        penalty_areas = {
            "left": {
                "mark": Point(pitch_left_x + penalty_spot_dx, center_y).buffer(penalty_mark_tol), # Left penalty mark area
                "goal_area": Polygon([
                    (pitch_left_x - goal_area_tol, center_y - goal_area_half_w - goal_area_tol), 
                    (pitch_left_x + goal_area_dx + goal_area_tol, center_y - goal_area_half_w - goal_area_tol), 
                    (pitch_left_x + goal_area_dx + goal_area_tol, center_y + goal_area_half_w + goal_area_tol), 
                    (pitch_left_x - goal_area_tol, center_y + goal_area_half_w + goal_area_tol)
                ]), # Left goal area with tolerance
                "penalty_area": Polygon([
                    (pitch_left_x - penalty_area_tol, center_y - penalty_area_half_w - penalty_area_tol), 
                    (pitch_left_x + penalty_area_dx + penalty_area_tol, center_y - penalty_area_half_w - penalty_area_tol), 
                    (pitch_left_x + penalty_area_dx + penalty_area_tol, center_y + penalty_area_half_w + penalty_area_tol), 
                    (pitch_left_x - penalty_area_tol, center_y + penalty_area_half_w + penalty_area_tol)        
                ]) # Left penalty area with tolerance
            },
            "right": {
                "mark": Point(pitch_right_x - penalty_spot_dx, center_y).buffer(penalty_mark_tol), # Right penalty mark area
                "goal_area": Polygon([
                    (pitch_right_x - goal_area_dx - goal_area_tol, center_y - goal_area_half_w - goal_area_tol), 
                    (pitch_right_x + goal_area_tol, center_y - goal_area_half_w - goal_area_tol), 
                    (pitch_right_x + goal_area_tol, center_y + goal_area_half_w + goal_area_tol), 
                    (pitch_right_x - goal_area_dx - goal_area_tol, center_y + goal_area_half_w + goal_area_tol) 
                ]), # Right goal area with tolerance
                "penalty_area": Polygon([
                    (pitch_right_x - penalty_area_dx - penalty_area_tol, center_y - penalty_area_half_w - penalty_area_tol), 
                    (pitch_right_x + penalty_area_tol, center_y - penalty_area_half_w - penalty_area_tol), 
                    (pitch_right_x + penalty_area_tol, center_y + penalty_area_half_w + penalty_area_tol), 
                    (pitch_right_x - penalty_area_dx - penalty_area_tol, center_y + penalty_area_half_w + penalty_area_tol)
                ]) # Right penalty area with tolerance
            }
        }

        for penalty_side, penalty_area in penalty_areas.items():
            penalty_mark = penalty_area["mark"]
            goal_area = penalty_area["goal_area"]
            penalty_area = penalty_area["penalty_area"]

            if penalty_side == "left":
                defending_gk_player_point = home_gk_player_point if home_direction > 0 else away_gk_player_point
                attacking_player_points = away_player_points if home_direction > 0 else home_player_points
            elif penalty_side == "right":
                defending_gk_player_point = away_gk_player_point if home_direction > 0 else home_gk_player_point
                attacking_player_points = home_player_points if home_direction > 0 else away_player_points
            else:
                raise ValueError(f"Unexpected penalty side: {penalty_side}. Frame: {ball_feats.index[0]}~{ball_feats.index[-1]}, Set Piece Index: {set_piece_idx}")

            if (
                penalty_mark.contains(ball_point) and
                goal_area.contains(defending_gk_player_point) and 
                sum(penalty_area.contains(p) for p in attacking_player_points.values()) == 1
            ):
                
                penalty_candidates = [pid for pid, p in attacking_player_points.items() if penalty_area.contains(p)] # Only one player is within the penalty area.
                player, control_score, candidate_players = build_candidates(penalty_candidates, score_scale=0.0)
                
                return pd.DataFrame([
                    {
                        "control_idx": set_piece_idx,
                        "player": player,
                        "type": "control",
                        "subtype": "penalty_kick",
                        "control_score": np.round(control_score, 3),
                        "candidate_players": candidate_players,
                    }
                ]
            )

        # 5. Goal Kick: 공이 골 에어리어 근처에 있고, 수비팀 골키퍼가 골 에어리어 안에 있으며, 공격팀 선수들이 페널티 에어리어 안에 없는 경우, 수비팀 골키퍼를 goal kick을 수행한 선수로 간주한다.
        for penalty_side, penalty_area in penalty_areas.items():      
            goal_area = penalty_area["goal_area"]
            penalty_area = penalty_area["penalty_area"]

            if penalty_side == "left":
                defending_gk_player, defending_gk_player_point = (home_gk_pid, home_gk_player_point) if home_direction > 0 else (away_gk_pid, away_gk_player_point)
                attacking_player_points = away_player_points if home_direction > 0 else home_player_points
            elif penalty_side == "right":
                defending_gk_player, defending_gk_player_point = (away_gk_pid, away_gk_player_point) if home_direction > 0 else (home_gk_pid, home_gk_player_point)
                attacking_player_points = home_player_points if home_direction > 0 else away_player_points
            else:
                raise ValueError(f"Unexpected penalty side: {penalty_side}. Frame: {ball_feats.index[0]}~{ball_feats.index[-1]}, Set Piece Index: {set_piece_idx}")

            if (
                goal_area.contains(ball_point) and
                goal_area.contains(defending_gk_player_point) and 
                sum(penalty_area.contains(p) for p in attacking_player_points.values()) == 0
            ):
                goal_kick_candidates = [defending_gk_player] # 골킥 상황에서는 수비 골키퍼만 후보로 고려한다.
                player, control_score, candidate_players = build_candidates(goal_kick_candidates, score_scale=0.0) 

                return pd.DataFrame([
                    {
                        "control_idx": set_piece_idx,
                        "player": player,
                        "type": "control",
                        "subtype": "goal_kick",
                        "control_score": np.round(control_score, 3),
                        "candidate_players": candidate_players,
                    }
                ])

        # 6) Free Kick: 공이 경기장 내 특정 위치에서 시작되고, 세트피스 상황에 해당하는 다른 유형(코너킥, 페널티킥, 골킥, 스로인)이 검출되지 않는 경우, 공과의 거리가 가장 가까운 선수를 free kick을 수행한 선수로 간주한다.
        free_kick_candidates = player_points.keys()
        player, control_score, candidate_players = build_candidates(free_kick_candidates, score_scale=5.0) # 5.0: Heuristic distance threshold for free kick candidate scoring.

        return pd.DataFrame([
            {
                "control_idx": set_piece_idx,
                "player": player,
                "type": "control",
                "subtype": "free_kick",
                "control_score": np.round(control_score, 3),
                "candidate_players": candidate_players,
            }
        ])

    def detect_out_by_distance(
            self, ball_feats: pd.DataFrame, ball_dists: pd.DataFrame = None
        ) -> pd.DataFrame:
        """
            본 알고리즘은 이벤트 라벨 없이 트래킹 데이터만을 이용하여 out, pause 이벤트를 휴리스틱하게 검출하는 방법이다.
            episode의 마지막 프레임은 dead ball 상황이므로 항상 마지막 프레임은 out 또는 pause로 검출된다고 가정한다. 
            따라서 episode의 마지막 프레임에서 pitch line과의 거리가 가까운 경우 out으로, 그렇지 않은 경우 pause로 검출한다.
        """

        player_dist_func = Postprocessor.linear_scoring_func(0, 3, increasing=False)

        out_records = []
        out_idx = ball_feats.index[-1]
        
        use_cols = ["OUT-L", "OUT-R", "OUT-B", "OUT-T"]
        window_index = ball_feats.loc[out_idx-3:out_idx+3].index

        player_dist_score = player_dist_func(ball_dists.loc[window_index, use_cols]).max()
        total_scores = player_dist_score

        control_score = total_scores.max()
        player = total_scores.idxmax()
        
        if pd.isna(control_score) or control_score < 0.5:
            out_records.append({
                "control_idx": out_idx,
                "player": "PAUSE",
                "type": "control",
                "subtype": "pause",
                "control_score": None,
                "candidate_players": ",".join(total_scores.sort_values(ascending=False).index[:3].tolist()), 
            })
        else:
            out_records.append({
                "control_idx": out_idx,
                "player": player,
                "type": "control",
                "subtype": "out",
                "control_score": np.round(control_score, 3),
                "candidate_players": ",".join(total_scores.sort_values(ascending=False).index[1:4].tolist()),
            })

        return pd.DataFrame(out_records)
    
    def finetune_ball_trace(
        self, traces: pd.DataFrame, carry_records: pd.DataFrame = None
    ) -> pd.DataFrame:
        output_cols = ["carrier", "ball_x", "ball_y", "ball_z"]
        output = pd.DataFrame(index=traces.index, columns=output_cols)
        output[output_cols[1:]] = output[output_cols[1:]].astype(float)
       
        # Reconstruct the ball trace
        for i in carry_records.index:
            start_idx = carry_records.at[i, "start_idx"]
            end_idx = carry_records.at[i, "end_idx"]
            carrier = carry_records.at[i, "player"]

            output.loc[start_idx:end_idx, "carrier"] = carrier

            if not carrier.startswith("OUT"):
                output.loc[start_idx:end_idx, "ball_x"] = traces.loc[start_idx:end_idx, f"{carrier}_x"]
                output.loc[start_idx:end_idx, "ball_y"] = traces.loc[start_idx:end_idx, f"{carrier}_y"]
                output.loc[start_idx:end_idx, "ball_z"] = traces.loc[start_idx:end_idx, self.BALL_z]
            elif carrier in ["OUT-L", "OUT-R"]:
                output.loc[start_idx:end_idx, "ball_x"] = traces[f"{carrier}_x"].iloc[0]
                output.loc[start_idx:end_idx, "ball_y"] = traces.loc[start_idx:end_idx, self.BALL_y].mean()
                output.loc[start_idx:end_idx, "ball_z"] = 0
            elif carrier in ["OUT-B", "OUT-T"]:
                output.loc[start_idx:end_idx, "ball_x"] = traces.loc[start_idx:end_idx, self.BALL_x].mean()
                output.loc[start_idx:end_idx, "ball_y"] = traces[f"{carrier}_y"].iloc[0]
                output.loc[start_idx:end_idx, "ball_z"] = 0
            else:
                raise ValueError(f"Invalid carrier: {carrier}")


        # carrier가 존재하지 않는 프레임은 결측치로 보간
        output[["ball_x", "ball_y"]] = output[["ball_x", "ball_y"]].interpolate(limit_direction="both")

        return output

    def run(self, method="ball_accel", max_accel=5, evaluate=False):
        if evaluate:
            n_frames = 0
            sum_pos_error = 0
            # correct_team_poss = 0
            # correct_player_poss = 0

        event_records_list = []

        for phase in tqdm(self.traces["phase"].unique(), desc="Postprocessing"):
            if phase == 0:
                continue
            phase_traces = self.traces[self.traces["phase"] == phase].copy()
            period_id = phase_traces["period_id"].iloc[0]
            
            players = phase_traces.dropna(axis=1).columns 
            players = [p.split('_')[0] for p in players if p.endswith("_x") and (p.startswith("H") or p.startswith("A"))]

            if method == "ball_accel":
                episodes = [e for e in phase_traces["episode"].unique() if e > 0]
                
                for episode in episodes:
                    ep_traces = phase_traces[phase_traces["episode"] == episode].copy()
                    #print(f"Processing Episode {episode} in Phase {phase}...")
                    if np.all(np.isnan(ep_traces[self.BALL_x])) or np.all(np.isnan(ep_traces[self.BALL_y])):
                        print(f"Episode {episode} in Phase {phase} has all NaN ball position values!")
                        continue

                    ball_feats = self.calc_ball_features(ep_traces)
                    # 트래킹 데이터 특성상 속도값이 굉장히 튀는 경우 제외
                    # if np.all(np.isnan(ball_feats["vx"])) or np.all(np.isnan(ball_feats["vy"])):
                    #     print(f"Episode {episode} in Phase {phase} has all NaN ball velocity values!")
                    #     continue
                        
                    ball_dists = self.calc_ball_dists(ep_traces, players)
                    control_records = self.detect_control_by_accel(ball_feats, ball_dists, max_accel=max_accel)
                    out_records = self.detect_out_by_distance(ball_feats, ball_dists)
                    set_piece_records = self.detect_set_piece_by_distance(ep_traces, ball_feats, control_records, players)

                    records = pd.concat(
                        [set_piece_records, control_records, out_records], ignore_index=True
                    ).sort_values(by="control_idx").reset_index(drop=True)

                    touch_records = self.generate_touch_records(records)
                    carry_records = self.generate_carry_records(records)   
                    kick_records = self.generate_kick_records(records)
                    out_records = self.generate_out_records(out_records)
       
                    kick_records = self.generate_shot_records(ep_traces, kick_records, players)
                    touch_records = self.generate_keeper_records(ep_traces, touch_records, kick_records, players)
                    
                    event_records = pd.concat(
                        [touch_records, carry_records, kick_records, out_records], ignore_index=True
                    ).sort_values(by=["start_idx", "end_idx"]).reset_index(drop=True)

                    event_records["period_id"] = period_id
                    event_records["phase"] = phase
                    event_records["episode"] = episode
                    event_records_list.append(event_records)
                    
                    ep_output = self.finetune_ball_trace(ep_traces, carry_records)
                    self.output.loc[ep_traces.index] = ep_output

                    if evaluate:
                        n_frames += ep_traces.shape[0]

                        error_x = (ep_output["ball_x"] - ep_traces["ball_x"]).values
                        error_y = (ep_output["ball_y"] - ep_traces["ball_y"]).values
                        sum_pos_error += np.sqrt((error_x**2 + error_y**2).astype(float)).sum()
                        if np.any(np.isnan(ep_output["ball_x"])) or np.any(np.isnan(ep_output["ball_y"])):
                            print(f"Episode {episode} in Phase {phase} has all NaN ball position values in output!")
                            print(f"n_frames: {n_frames}, error_x: {error_x}, error_y: {error_y}, sum_pos_error: {sum_pos_error}")
                            exit()
                        if np.any(np.isnan(ep_traces["ball_x"])) or np.any(np.isnan(ep_traces["ball_y"])):
                            print(f"Episode {episode} in Phase {phase} has all NaN ball position values in traces!")
                            print(f"n_frames: {n_frames}, error_x: {error_x}, error_y: {error_y}, sum_pos_error: {sum_pos_error}")
                            exit()

                        # pposs_pred = ep_output["carrier"].fillna(method="bfill").fillna(method="ffill")
                        # pposs_target = ep_traces["player_poss"].fillna(method="bfill").fillna(method="ffill")
                        # correct_player_poss += (pposs_pred == pposs_target).astype(int).sum()

                        # tposs_pred = pposs_pred.apply(lambda x: x[0])
                        # tposs_target = pposs_target.apply(lambda x: x[0])
                        # correct_team_poss += (tposs_pred == tposs_target).astype(int).sum()

                self.output.loc[phase_traces.index, ["ball_x", "ball_y"]] = self.output.loc[
                    phase_traces.index, ["ball_x", "ball_y"]
                ].interpolate(limit_direction="both")
            else:
                raise ValueError(f"Not Exists Method: {method}")

        self.event_records = pd.concat(event_records_list, ignore_index=True)
        self.event_records["start_frame"] = self.traces.loc[self.event_records["start_idx"], "frame_id"].values
        self.event_records["end_frame"] = self.traces.loc[self.event_records["end_idx"], "frame_id"].values
        self.event_records[["start_x", "start_y", "start_z"]] = self.traces.loc[self.event_records["start_idx"], [self.BALL_x, self.BALL_y, self.BALL_z]].values
        self.event_records[["end_x", "end_y", "end_z"]] = self.traces.loc[self.event_records["end_idx"], [self.BALL_x, self.BALL_y, self.BALL_z]].values

        if evaluate and n_frames > 0:
            stats = {"n_frames": n_frames}
            stats["sum_pos_error"] = sum_pos_error
            # stats["correct_player_poss"] = correct_player_poss
            # stats["correct_team_poss"] = correct_team_poss
            return stats
        else:
            return None

    @staticmethod
    def linear_scoring_func(min_input: float, max_input: float, increasing=False):
        assert min_input < max_input

        def func(x: float) -> float:
            if increasing:
                return (x - min_input) / (max_input - min_input)
            else:
                return 1 - (x - min_input) / (max_input - min_input)

        return lambda x: np.maximum(0, np.minimum(1, func(x)))

    @staticmethod
    def plot_speed_and_accel_curves(frames: pd.Series, ball_traces: pd.DataFrame, event_records: pd.DataFrame):
        plt.rcParams.update({"font.size": 15})
        fig, axes = plt.subplots(2, 1)
        fig.set_facecolor("w")
        fig.set_size_inches(15, 10)
        fig.subplots_adjust(right=0.78)

        type_colors = {
            "carry":   "#27AE60",  # green-teal
            "kick":    "#F2994A",  # orange
            "out":     "#6D597A",  # muted purple-gray
            "control": "#2F80ED",  # blue,

            # 세트피스 (색상 다른 버전)
            # "free_kick":    "#1F3A5F",  # deep navy
            # "corner_kick":  "#0B6E4F",  # deep teal
            # "throw_in":     "#5A5A00",  # olive
            # "penalty_kick": "#5C1A1B",  # wine red
            # "goal_kick":    "#4B0082",  # indigo

            # 세트피스 (색상 통일 버전)
            "free_kick":    "#4B0082",  # blue
            "corner_kick":  "#4B0082",  # blue
            "throw_in":     "#4B0082",  # blue
            "penalty_kick": "#4B0082",  # blue
            "goal_kick":    "#4B0082",  # blue  
        }

        # Existing rendering (kept as requested)
        for i in event_records.index:
            event_type = event_records.at[i, "type"]

            start_frame = (event_records.at[i, "start_frame"])
            end_frame = (event_records.at[i, "end_frame"])
            axes[0].axvspan(start_frame, end_frame, alpha=0.4, color=type_colors.get(event_type), linewidth=2.5)
            axes[1].axvspan(start_frame, end_frame, alpha=0.4, color=type_colors.get(event_type), linewidth=2.5)

        start_margin = 5
        end_margin = 5
        for i in event_records.index:
            event_type = event_records.at[i, "type"]
            start_frame = event_records.at[i, "start_frame"]
            end_frame = event_records.at[i, "end_frame"]
            
            start_frame_speed = ball_traces[ball_traces["frame"] == start_frame]["speed"].values[0]
            start_frame_accel = ball_traces[ball_traces["frame"] == start_frame]["accel"].values[0]
            end_frame_speed = ball_traces[ball_traces["frame"] == end_frame]["speed"].values[0]
            end_frame_accel = ball_traces[ball_traces["frame"] == end_frame]["accel"].values[0]
            
            axes[0].text(
                start_frame, start_frame_speed-start_margin, f"{start_frame}",
                ha="center", va="top", zorder=4,
                fontsize=20, color=type_colors.get(event_type)
            )
            axes[0].text(
                end_frame, end_frame_speed+end_margin, f"{end_frame}",
                ha="center", va="top", zorder=4,
                fontsize=20, color=type_colors.get(event_type)
            )
            axes[1].text(
                start_frame, start_frame_accel-start_margin, f"{start_frame}",
                ha="center", va="top", zorder=4,
                fontsize=20, color=type_colors.get(event_type)
            )
            axes[1].text(
                end_frame, end_frame_accel+end_margin, f"{end_frame}",
                ha="center", va="top", zorder=4,
                fontsize=20, color=type_colors.get(event_type)
            )

        xmin = frames.iloc[0]
        xmax = frames.iloc[-1]
        axes[0].set(xlim=(xmin, xmax), ylim=(0, 30))
        axes[1].set(xlim=(xmin, xmax), ylim=(-50, 50))
        axes[0].plot(frames, ball_traces["speed"], color="black", linestyle="-", linewidth=2)
        axes[1].plot(frames, ball_traces["accel"], color="black", linestyle="-", linewidth=2)
        axes[0].set_ylabel("Speed [m/s]")
        axes[1].set_ylabel("Acceleration [m/s²]")
        axes[1].set_xlabel("Frame [25fps]")

        ticks = np.linspace(xmin, xmax, 15)
        axes[0].set_xticks(ticks)
        axes[1].set_xticks(ticks)

        event_handles = []
        for etype, color in type_colors.items():
            event_handles.append(Patch(facecolor=color, alpha=0.5, label=etype))

        speed_handle = Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="speed")
        accel_handle = Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="accel")

        # 두 축 공통 범례를 figure 기준으로 우측 바깥에 한 번만 표시한다.
        shared_handles = event_handles + [speed_handle, accel_handle]
        fig.legend(
            handles=shared_handles,
            loc="center left",
            bbox_to_anchor=(0.81, 0.5),
            fontsize=25,
            ncol=1,
            frameon=True,
        )

        axes[0].grid()
        axes[1].grid()
        
    @staticmethod
    def detect_false_poss_segments(traces: pd.DataFrame) -> pd.DataFrame:
        true_poss = traces["player_poss"].fillna(method="bfill").fillna(method="ffill")
        pred_poss = traces["pred_poss"]

        false_idxs = true_poss[true_poss != pred_poss].reset_index()["index"]
        time_diffs = pd.Series(false_idxs.diff().fillna(10).values, index=false_idxs)
        segment_ids = (time_diffs > 3).astype(int).cumsum().rename("segment_id").reset_index()

        start_idxs = segment_ids.groupby("segment_id")["index"].first().rename("start_idx")
        end_idxs = segment_ids.groupby("segment_id")["index"].last().rename("end_idx")
        false_segments = pd.concat([start_idxs, end_idxs], axis=1)

        false_segments["miss"] = False
        false_segments["false_alarm"] = False

        for i in false_segments.index:
            i0 = false_segments.at[i, "start_idx"]
            i1 = false_segments.at[i, "end_idx"]

            true_players = true_poss.loc[i0:i1].unique()
            pred_players = pred_poss.loc[i0:i1].unique()
            true_players_ext = true_poss.loc[i0 - 10 : i1 + 10].unique()
            pred_players_ext = pred_poss.loc[i0 - 10 : i1 + 10].unique()

            false_segments.at[i, "miss"] = len(set(true_players) - set(pred_players_ext)) != 0
            false_segments.at[i, "false_alarm"] = len(set(pred_players) - set(true_players_ext)) != 0

        return false_segments

    @staticmethod
    def plot_poss_and_error_curves(
        traces: pd.DataFrame,
        poss_scores: pd.DataFrame,
        pp_output: pd.DataFrame = None,
        mark_turns: bool = False,
        thres_accel: float = 5,
    ) -> animation.FuncAnimation:
        FRAME_DUR = 30
        MAX_DIST = 20

        nn_pos_error_xy = traces[["ball_x", "ball_y"]] - traces[["pred_ball_x", "pred_ball_y"]].values
        nn_pos_errors = nn_pos_error_xy.apply(np.linalg.norm, axis=1)
        if pp_output is not None:
            pp_pos_error_xy = traces[["ball_x", "ball_y"]] - pp_output[["ball_x", "ball_y"]].values
            pp_pos_errors = pp_pos_error_xy.apply(np.linalg.norm, axis=1)

        poss_cols = [p for p in poss_scores.dropna(axis=1).columns if p[0] in ["A", "B", "O"]]
        poss_dict = dict(zip(poss_cols, np.arange(len(poss_cols))))
        poss_dict["GOAL-L"] = len(poss_cols) - 4
        poss_dict["GOAL-R"] = len(poss_cols) - 3

        true_poss = traces["player_poss"].dropna().map(poss_dict)
        nn_pred_poss = traces["pred_poss"].map(poss_dict)
        if pp_output is not None:
            pp_pred_poss = pp_output["carrier"].dropna().map(poss_dict)

        plt.rcParams.update({"font.size": 15})
        fig, axes = plt.subplots(3, 1)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0.05)
        fig.set_size_inches(15, 20)

        times = traces["time"].values
        t0 = int(times[0] - 0.1)

        axes[0].plot(times[true_poss.index], true_poss, color="tab:blue", marker="o", label="True")
        axes[0].plot(times, nn_pred_poss, color="orangered", marker="o", label="NN output")
        if pp_output is not None:
            axes[0].plot(times[pp_pred_poss.index], pp_pred_poss, color="darkgreen", marker="o", label="PP output")

        axes[0].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-1, len(poss_cols)))
        axes[0].set_xticklabels([])
        axes[0].set_yticks(ticks=np.arange(len(poss_cols)), labels=poss_cols)
        axes[0].set_ylabel("Ball possessor", fontdict={"size": 20})
        axes[0].grid()
        axes[0].legend(loc="upper right")

        n_players = (len(poss_cols) - 4) // 2
        base_cmaps = ["hot_r", "winter_r", "Greys_r"]
        colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.1, 0.9, n_players)) for name in base_cmaps])
        poss_cols = poss_cols[:n_players] + poss_cols[-4:-2] + poss_cols[n_players:-4] + poss_cols[-2:]
        for p in poss_cols:
            axes[1].plot(times, poss_scores[p], label=p, color=colors[poss_dict[p]])

        if mark_turns:
            ball_features = Postprocessor.calc_ball_features(traces)
            accels = ball_features[["accel"]].copy()
            for k in np.arange(2) + 1:
                accels[f"prev{k}"] = accels["accel"].shift(k, fill_value=0)
                accels[f"next{k}"] = accels["accel"].shift(-k, fill_value=0)

            max_flags = (accels["accel"] == accels.max(axis=1)) & (accels["accel"] > thres_accel)
            max_idxs = accels[max_flags].index.tolist()
            max_times = traces.loc[max_idxs, "time"]
            max_scores = poss_scores.loc[max_idxs, poss_cols].max(axis=1)
            axes[1].scatter(max_times, max_scores.clip(0, 1), s=200, c="tab:red", marker="^")

            min_flags = (accels["accel"] == accels.min(axis=1)) & (accels["accel"] < -thres_accel)
            min_idxs = accels[min_flags].index.tolist()
            min_times = traces.loc[min_idxs, "time"]
            min_scores = poss_scores.loc[min_idxs, poss_cols].max(axis=1)
            axes[1].scatter(min_times, min_scores.clip(0, 1), s=200, c="tab:blue", marker="v")

        axes[1].set(xlim=(t0, t0 + FRAME_DUR), ylim=(0, 1.05))
        axes[1].set_xticklabels([])
        axes[1].set_ylabel("Possession probability", fontdict={"size": 20})
        axes[1].grid(which="major", axis="both")
        axes[1].legend(loc="upper right", ncols=2)

        axes[2].plot(times, nn_pos_errors, color="orangered", label="NN output")
        if pp_output is not None:
            axes[2].plot(times, pp_pos_errors, color="tab:green", label="PP output")
            axes[2].legend(loc="upper right")

        axes[2].set(xlim=(t0, t0 + FRAME_DUR), ylim=(0, MAX_DIST))
        axes[2].set_xlabel("Time [s]", fontdict={"size": 20})
        axes[2].set_ylabel("Position error", fontdict={"size": 20})
        axes[2].grid()

        false_segments = Postprocessor.detect_false_poss_segments(traces)
        for i in tqdm(false_segments.index):
            start_time = traces.at[false_segments.at[i, "start_idx"], "time"] - 0.05
            end_time = traces.at[false_segments.at[i, "end_idx"], "time"] + 0.05

            miss = false_segments.at[i, "miss"]
            false_alarm = false_segments.at[i, "false_alarm"]

            if miss and false_alarm:
                color = "tab:red"
            elif miss:
                color = "tab:blue"
            elif false_alarm:
                color = "tab:orange"
            else:
                color = "tab:gray"

            axes[0].axvspan(start_time, end_time, alpha=0.3, color=color)
            axes[1].axvspan(start_time, end_time, alpha=0.3, color=color)
            axes[2].axvspan(start_time, end_time, alpha=0.3, color=color)

        def animate(i):
            for ax in axes:
                ax.set_xlim(10 * i, 10 * i + FRAME_DUR)

        frames = (len(traces) - 10 * FRAME_DUR) // 100 + 1
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=500)
        plt.close(fig)

        return anim
