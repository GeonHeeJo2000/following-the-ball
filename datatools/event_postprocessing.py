"""
이벤트 데이터를 generate_visualization.py가 요구하는 형식으로 변환한다.

필수 출력 컬럼:
  - player_id  : "home_N" / "away_N"
  - event_type : "kick" | "control"
  - x          : pitch x 좌표 (0~105)
  - y          : pitch y 좌표 (0~68)
"""
from __future__ import annotations

import pandas as pd

# GT event_kind → event_type 매핑
_GT_KICK_KINDS = {
    "Pass", "Cross", "ShotAtGoal", "FreeKick",
    "CornerKick", "GoalKick", "ThrowIn", "KickOff",
}
_GT_CONTROL_KINDS = {"OtherBallAction", "BallClaiming"}

# xgb pred_label → event_type 매핑
_XGB_KICK_LABELS = {"pass", "cross", "shot"}


def _from_gt(events: pd.DataFrame) -> pd.DataFrame:
    """raw_gt_synced 포맷 → 표준 events DataFrame."""
    df = events.copy()

    # event_type 매핑
    df["event_type"] = df["event_kind"].map(
        lambda k: "kick" if k in _GT_KICK_KINDS else ("control" if k in _GT_CONTROL_KINDS else None)
    )
    df = df[df["event_type"].notna()].copy()

    # x/y 이미 존재
    df = df.dropna(subset=["x", "y", "player_id"])
    return df[["player_id", "event_type", "x", "y"]].reset_index(drop=True)


def _from_xgb(events: pd.DataFrame) -> pd.DataFrame:
    """xgb_predictions 포맷 → 표준 events DataFrame."""
    df = events.copy()

    df["event_type"] = df["pred_label"].map(
        lambda l: "kick" if l in _XGB_KICK_LABELS else None
    )
    df = df[df["event_type"].notna()].copy()

    df["player_id"] = df["loss_player"]
    df["x"] = df["ball_x"]
    df["y"] = df["ball_y"]
    df = df.dropna(subset=["x", "y", "player_id"])
    return df[["player_id", "event_type", "x", "y"]].reset_index(drop=True)


def prepare_events(
    events: pd.DataFrame,
    tracking: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    events DataFrame을 heatmap 생성에 필요한 표준 형식으로 변환한다.

    GT (raw_gt_synced) 판별: 'event_kind' 컬럼 존재 여부
    XGB (xgb_predictions) 판별: 'pred_label' 컬럼 존재 여부
    """
    if events is None or events.empty:
        return pd.DataFrame(columns=["player_id", "event_type", "x", "y"])

    if "event_kind" in events.columns:
        return _from_gt(events)
    elif "pred_label" in events.columns:
        return _from_xgb(events)
    else:
        raise ValueError(
            "events에 'event_kind'(GT) 또는 'pred_label'(XGB) 컬럼이 필요합니다."
        )
