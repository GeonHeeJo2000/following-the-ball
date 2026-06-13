# Preprocessing parameters
TRACKING_FPS = 25
PERIOD_START_FRAME = {1: 10000, 2: 100000}

# DFL event_kind -> set of acceptable spadl event_type values for a
# "player_label" (strict) match. Anything outside this set with the same
# player_id falls back to "player_only".
DFL_TO_SPADL = {
    "Pass": {"pass"},
    "Cross": {"cross"},
    "ShotAtGoal": {"shot", "shot_block", "goal", "shot_freekick",
                   "shot_penalty"},
    "ThrowIn": {"throw_in"},
    "FreeKick": {"freekick_short", "freekick_crossed", "shot_freekick"},
    "CornerKick": {"corner_short", "corner_crossed"},
    "GoalKick": {"goalkick"},
    "KickOff": {"kickoff"},
    "TacklingGame": {"tackle", "dispossessed", "interception",
                     "ball_recovery"},
    "BallClaiming": {"control", "interception", "ball_recovery"},
    "OtherBallAction": {"clearance", "bad_touch", "control", "pass"},
}

# Two-tier matching window. A same-player candidate is accepted if
#   - its event_type is in the DFL->spadl set AND it lies within
#     KIND_MATCH_WINDOW_S of the raw timestamp, OR
#   - its event_type does NOT match but it lies within the tighter
#     ANY_MATCH_WINDOW_S. The looser window grabs clearly-corresponding
#     events reported up to ~10s late; the tighter window for
#     non-kind-matches prevents grabbing an unrelated control/tackle 7s
#     away just because it shares the player.
KIND_MATCH_WINDOW_S = 10.0
ANY_MATCH_WINDOW_S = 5.0

# Training Dataset parameters
PITCH_X, PITCH_Y = 105.0, 68.0
PA_X = 16.5
PA_Y_MIN = (PITCH_Y - 40.3) / 2
PA_Y_MAX = PITCH_Y - PA_Y_MIN
TRACKING_FPS = 25
GOAL_HALF_W = 3.66

KICK_LABELS = ["none", "pass", "cross", "shot"]
KICK_LABEL_TO_IDX = {label: idx for idx, label in enumerate(KICK_LABELS)}

KICK_FEATURE_COLS = [
    "ball_x", "ball_y", "ball_z",
    "ball_vx", "ball_vy", "ball_speed", "ball_accel",
    "ball_z_max_1s", "delta_dist_to_goal_1s",
    "dist_to_goal", "cos_to_goal", "sideline_dist", "rel_x",
    "in_attacking_half", "in_shot_zone", "in_cross_zone", "in_attacking_pa",
    "lp_x", "lp_y", "lp_speed",
    "nearest_opp_dist", "n_opps_in_5m", "n_attackers_in_pa",
    "time_to_gain", "gain_in_attacking_pa", "same_team_gain", "opp_team_gain",
    "is_set_piece",
    "team_in_cone15", "team_in_cone30", "opp_in_cone30",
    "nearest_team_along", "nearest_opp_along",
    "past_speed_mean", "past_speed_max", "past_accel_max", "past_z_max",
    "future_speed_max", "future_z_max_2s", "future_air_frames",
    "ball_dir_stability",
    "time_since_prev_loss", "time_since_prev_same",
    "rcv_speed", "rcv_align", "rcv_pressure",
    "traj_curvature",
    "sp_throw_in", "sp_corner", "sp_freekick",
    "sp_goalkick", "sp_kickoff", "sp_penalty",
]

SET_PIECE_LABELS:   list[str] = ["throw_in", "goal_kick", "corner_kick", "free_kick", "kickoff", "penalty_kick"]
