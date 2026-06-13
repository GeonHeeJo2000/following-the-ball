"""autoevent/helpers.py — 공통 헬퍼 함수 모음.

각 detector 클래스에서 반복되던 좌표 계산·선수 판별 로직을 순수 함수로 제공.
"""
from __future__ import annotations

from fnmatch import fnmatch

import numpy as np
import pandas as pd

from models.AutoEvent.config import (
    PITCH_X,
    PITCH_Y,
    PENALTY_AREA_X_MAX,
    PENALTY_AREA_Y_MIN,
    PENALTY_AREA_Y_MAX,
    GOAL_POST_Y_MIN,
    GOAL_POST_Y_MAX,
    SHOT_ON_OFF_TOL,
    SHOT_ZONE_X,
    CROSS_ZONE_Y,
    CROSS_ZONE_X_MIN,
    GK_IN_PA_X_TOL,
)


# ---------------------------------------------------------------------------
# 선수 목록 추출
# ---------------------------------------------------------------------------

def get_players(trk: pd.DataFrame) -> list[str]:
    """트래킹 DataFrame에서 선수 ID 목록을 반환 (home_*/away_* 패턴)."""
    x_cols = [
        col for col in trk.columns
        if fnmatch(col, "home_*_x") or fnmatch(col, "away_*_x")
    ]
    return sorted({col[:-2] for col in x_cols})


# ---------------------------------------------------------------------------
# ID / 좌표 유틸
# ---------------------------------------------------------------------------

def team_of(player_id: str) -> str:
    """선수 ID에서 팀명 추출 ('home_7' → 'home')."""
    return str(player_id).split("_")[0]


def player_xy(row: pd.Series, player: str) -> tuple[float, float]:
    """행(row)에서 선수의 (x, y) 좌표를 float 튜플로 반환. 없으면 (nan, nan)."""
    return (
        float(row.get(f"{player}_x", np.nan)),
        float(row.get(f"{player}_y", np.nan)),
    )


# ---------------------------------------------------------------------------
# 공격 방향 / 골라인
# ---------------------------------------------------------------------------

def attacking_goal_x(team: str) -> float:
    """팀이 공격하는 골라인 x 좌표 (home → PITCH_X, away → 0.0)."""
    return PITCH_X if team == "home" else 0.0


# ---------------------------------------------------------------------------
# 구역 판별
# ---------------------------------------------------------------------------

def in_shot_zone(x: float, y: float, team: str) -> bool:
    """볼 위치가 해당 팀 기준 슈팅 존 안인지."""
    if np.isnan(x) or np.isnan(y):
        return False
    rel_x = x if team == "home" else PITCH_X - x
    return rel_x >= SHOT_ZONE_X and PENALTY_AREA_Y_MIN <= y <= PENALTY_AREA_Y_MAX


def in_cross_zone(x: float, y: float, team: str) -> bool:
    """볼 위치가 해당 팀 기준 크로스 존 안인지."""
    if np.isnan(x) or np.isnan(y):
        return False
    rel_x = x if team == "home" else PITCH_X - x
    side = (y < CROSS_ZONE_Y) or (y > PITCH_Y - CROSS_ZONE_Y)
    return (rel_x >= CROSS_ZONE_X_MIN) and side


def in_attacking_pa(px: float, py: float, attacking_team: str) -> bool:
    """주어진 좌표가 attacking_team의 공격 방향 PA 안인지."""
    if np.isnan(px) or np.isnan(py):
        return False
    if not (PENALTY_AREA_Y_MIN <= py <= PENALTY_AREA_Y_MAX):
        return False
    if attacking_team == "home":
        return px >= PITCH_X - PENALTY_AREA_X_MAX
    return px <= PENALTY_AREA_X_MAX


def in_defending_pa(px: float, py: float, defending_team: str) -> bool:
    """주어진 좌표가 defending_team의 수비 PA 안인지."""
    return in_attacking_pa(px, py, "away" if defending_team == "home" else "home")


# ---------------------------------------------------------------------------
# GK 관련
# ---------------------------------------------------------------------------

def detect_gks(tracking: pd.DataFrame, players: list[str], sample_frames: int = 125) -> set[str]:
    """전반 첫 alive 구간(기본 5초=125프레임)의 중앙값 x좌표로 GK를 추정.

    home GK: home 선수 중 중앙값 x 최솟값 (자기 골라인 x=0 근처)
    away GK: away 선수 중 중앙값 x 최댓값 (자기 골라인 x=PITCH_X 근처)
    단일 프레임 대신 여러 프레임 중앙값을 사용해 트래킹 노이즈에 강건하게 처리.
    """
    alive = tracking[tracking["ball_state"] == "alive"]
    sample = alive.head(sample_frames)
    gk_ids: set[str] = set()

    for t in ("home", "away"):
        team_players = [p for p in players if p.split("_")[0] == t]
        xs: dict[str, float] = {}
        for p in team_players:
            col = f"{p}_x"
            if col in sample.columns:
                vals = sample[col].dropna()
                if not vals.empty:
                    xs[p] = float(vals.median())
        if not xs:
            continue
        gk = min(xs, key=lambda p: xs[p]) if t == "home" else max(xs, key=lambda p: xs[p])
        gk_ids.add(gk)

    return gk_ids


def gk_in_pa(row: pd.Series, gk_player: str) -> bool:
    """GK가 자기 PA 안에 있는지 (GK_IN_PA_X_TOL 포함)."""
    px, py = player_xy(row, gk_player)
    if np.isnan(px):
        return False
    if not (PENALTY_AREA_Y_MIN <= py <= PENALTY_AREA_Y_MAX):
        return False
    t = team_of(gk_player)
    if t == "home":
        return px <= PENALTY_AREA_X_MAX + GK_IN_PA_X_TOL
    return px >= PITCH_X - PENALTY_AREA_X_MAX - GK_IN_PA_X_TOL


def attacker_in_pa(
    row: pd.Series,
    attacking_team: str,
    players: list[str],
    gk_ids: set[str],
) -> bool:
    """attacking_team 소속 필드 플레이어(GK 제외) 중 상대 PA 안에 1명 이상 있으면 True."""
    for pid in players:
        if team_of(str(pid)) != attacking_team:
            continue
        if str(pid) in gk_ids:
            continue
        px, py = player_xy(row, str(pid))
        if np.isnan(px) or np.isnan(py):
            continue
        if in_attacking_pa(px, py, attacking_team):
            return True
    return False


# ---------------------------------------------------------------------------
# 볼 방향
# ---------------------------------------------------------------------------

def ball_toward_goal(row: pd.Series, team: str) -> bool:
    """공이 나가는 방향이 공격 골문 쪽을 향하는지."""
    dx = row.get("ball_dir_out_x")
    dy = row.get("ball_dir_out_y")
    if dx is None or dy is None or pd.isna(dx) or pd.isna(dy):
        return False
    bx = row.get("ball_x", np.nan)
    if np.isnan(float(bx)):
        return False
    goal_x = attacking_goal_x(team)
    return float(dx) * (goal_x - float(bx)) > 0


def shot_on_target(row: pd.Series, team: str) -> bool:
    """공이 골포스트 범위 안을 향하는지 (on target 판별)."""
    dx = row.get("ball_dir_out_x")
    dy = row.get("ball_dir_out_y")
    bx = row.get("ball_x", np.nan)
    by = row.get("ball_y", np.nan)
    if any(pd.isna(v) for v in [dx, dy, bx, by]):
        return False
    dx, dy, bx, by = float(dx), float(dy), float(bx), float(by)
    if dx == 0:
        return False
    goal_x = attacking_goal_x(team)
    t = (goal_x - bx) / dx
    if t <= 0:
        return False
    hit_y = by + t * dy
    return (GOAL_POST_Y_MIN - SHOT_ON_OFF_TOL) <= hit_y <= (GOAL_POST_Y_MAX + SHOT_ON_OFF_TOL)
