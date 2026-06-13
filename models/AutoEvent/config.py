PITCH_X = 105.0  # unit: meters
PITCH_Y = 68.0  # unit: meters

TRACKING_FPS = 25  # 트래킹 데이터 프레임 레이트

# Possession detection hyperparameters
# 논문 Sec 3 권장: Provider A (Track160=DFL)=0.5, Provider B/C=1.0
# 그러나 elastic이 Savitzky-Golay 평활화한 tracking.parquet 위에서는
# 0.5m가 너무 빠듯해 Recall이 30% 가까이 하락 → 1.0m가 우리 파이프라인에서 최적
R_PZ = 1.5     # possession zone radius (m) — relaxed from 1.0 to capture body-extension touches (foot reach, headers, smoothing noise)
R_DZ = 1.5     # duel zone radius (m)
EPS_S = 0.1    # 볼 최소 이동 거리 (loss 감지, 0.1m = 10cm)
MIN_SEQ_LENGTH = 3  # possession 최소 프레임 수 (짧은 순간 접촉 제거)
EPS_THETA = 0.9848  # 볼 방향 변화 임계값 (gain 감지, cos(10°))
EPS_V = 0.2    # 볼 속도 변화 임계값 (gain 감지, 5m/s @25fps = 0.2m/frame)
SG_WINDOW = 7    # Savitzky-Golay smoothing window length
SG_POLYORDER = 2 # Savitzky-Golay polynomial order


# Set piece detection hyperparameters

# kickoff
CENTER_X = PITCH_X / 2
CENTER_Y = PITCH_Y / 2
KICKOFF_X_TOL = 2.0  # 중앙에서 얼마나 떨어져 있어도 킥오프로 간주할지 (m) e_k1
KICKOFF_Y_TOL = 2.0
HALF_TOL = 5.0  # 킥오프 판단 시 상대 진영에 있는 선수 허용 범위 (m)


# penalty kick
PENALTY_MARK_X = 11.0  # 페널티 마크의 x 좌표 (골라인에서부터의 거리, m)
PENALTY_MARK_TOL = 2.0  # 볼이 페널티 마크에서 얼마나 떨어져 있어도 페널티킥으로 간주할지 (m) ep_2
PENALTY_GK_X_TOL = 2.0  # GK가 골라인에서 얼마나 떨어져 있어도 GK로 간주 (m) ep_1
PENALTY_GK_Y_TOL = 2.0  # GK가 골라인 중앙에서 상하로 얼마나 떨어져 있어도 GK로 간주 (m) ep_1
PENALTY_OTHER_PLAYERS_TOL = 4.0  # 다른 선수가 페널티박스에서 떨어져야 할 거리 (m) ep_3
GOAL_X_TOL = 3.0  # 골라인 근처 판단 기준 (m)
GOAL_Y_TOL = 3.0

LEFT_GOAL_X = 0.0  # 왼쪽 골라인 x 좌표
RIGHT_GOAL_X = PITCH_X  # 오른쪽 골라인 x 좌표
PENALTY_AREA_X_MIN = 0.0
PENALTY_AREA_X_MAX = 16.5
PENALTY_AREA_Y_MIN = (PITCH_Y - 40.3) / 2
PENALTY_AREA_Y_MAX = PITCH_Y - PENALTY_AREA_Y_MIN

# goal kick
GOAL_AREA_X = 5.5  # 골 에어리어 크기 (m)
GOAL_AREA_Y = 18.3 / 2 # 골 에어리어 세로 크기(m) center 기준 위로 9.15m, 아래로 9.15m 
GOAL_AREA_X_TOL = 0.5  # 골 에어리어 근처 판단 기준 (m) — 논문: 0.5m
GOAL_AREA_Y_TOL = 0.5  # 골 에어리어 근처 판단 기준 (m) — 논문: 0.5m

# corner kick
CORNER_TOL = 3.0  # 코너킥 근처 판단 기준 (m)

# throw-in
THROW_IN_TOL = 0.5  # 스로인 근처 판단 기준 (m) — 논문: 0.5m

# fallback set piece detection (Issues 1, 2 & 3, paper S1)
FALLBACK_SCAN_STEP = 5  # dead ball 구간 스캔 시 프레임 샘플링 간격 (25fps 기준 0.2s)
EPS_F = 1.0   # Issue 1: first player in control must move more than this (m) to confirm execution frame
ISSUE1_SCAN_FRAMES = 75  # Issue 1: first_inplay_pos 이후 최대 탐색 프레임 수 (75 = 3초 @ 25fps)

# open play event detection — set piece 원인 loss 역방향 탐색 윈도우
SP_LOSS_MAP_WINDOW = 5  # set piece 시작점 기준 역방향 탐색 시간(초)
THROW_IN_BALL_Z_MIN = 1.5  # 첫 alive 프레임 ball_z > 이 값 → ThrowIn 강제. 84% TP, ~3% FP

# open play event detection
GOAL_POST_Y_MIN = (PITCH_Y - 7.32) / 2   # 골포스트 안쪽 y 최솟값
GOAL_POST_Y_MAX = PITCH_Y - GOAL_POST_Y_MIN  # 골포스트 안쪽 y 최댓값
SHOT_ON_OFF_TOL = 0.25                    # shot on/off target 판별 골포스트 바깥 허용 범위 (m)

# shot / cross zone x
SHOT_ZONE_X = 75.0  # 슈팅 존 x 좌표 (m)

# cross zone: 공격 방향 기준 사이드라인 쪽 (PITCH_Y 기준 좌우 각 CROSS_ZONE_Y 이내)
CROSS_ZONE_Y = 18.3  # 페널티 에어리어 y폭 바깥 (m) — 양쪽 사이드라인 쪽
CROSS_ZONE_X_MIN = 75.0  # 공격 방향으로 크로스 가능한 x 시작점 (m)

GK_IN_PA_X_TOL = 3.0  # GK가 페널티 에어리어 안에 있는지 판단 여유 (m)










SPADL_TYPES = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "second_take_on",
    "foul",
    "tackle",
    "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "bad_touch",
    "goalkick",
    "shot_block",  # new, pass-like
    "ball_recovery",  # new, incoming
    "keeper_sweeper",  # new, incoming
    "dispossessed",  # new, minor
]
SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]

# Event categories and parameters for ELASTIC
PASS_LIKE_OPEN = ["pass", "cross", "shot", "clearance", "keeper_punch", "shot_block"]
SET_PIECE_OOP = ["throw_in", "goalkick", "corner_short", "corner_crossed"]
SET_PIECE = SET_PIECE_OOP + ["freekick_short", "freekick_crossed", "shot_freekick", "shot_penalty"]
INCOMING = ["interception", "keeper_save", "keeper_claim", "keeper_pick_up", "keeper_sweeper", "ball_recovery"]
MINOR = ["tackle", "take_on", "second_take_on", "foul", "bad_touch", "dispossessed"]

TIME_KICKOFF = 5  # Stats Perform: 5, Sportec: 5 (seconds)
TIME_PASS_LIKE_OPEN = 10  # Stats Perform: 5, Sportec: 10 (seconds)
TIME_SET_PIECE = 15  # Stats Perform: 15, Sportec: 15 (seconds)
TIME_INCOMING = 10  # Stats Perform: 5, Sportec: 10 (seconds)
TIME_MINOR = 10  # Stats Perform: 5, Sportec: 10 (seconds)
FRAME_DELAY_START = 0  # Stats Perform: 0, Sportec: -1 (seconds)

# Additional event categories and parameters for ETSY
BAD_TOUCH = ["bad_touch"]
FAULT_LIKE = ["foul", "tackle"]
NOT_HANDLED = ["take_on", "second_take_on", "dispossessed"]

TIME_BAD_TOUCH = 5
TIME_FAULT_LIKE = 5

EVENT_COLS = [
    "frame_id",
    "period_id",
    "synced_ts",
    "utc_timestamp",
    "player_id",
    "object_id",
    "player_name",
    "advanced_position",
    "spadl_type",
    "success",
    "offside",
    "expected_goal",
]
NEXT_EVENT_COLS = ["next_player_id", "next_type", "receiver_id", "receive_frame_id", "receive_ts"]
