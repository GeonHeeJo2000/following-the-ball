"""autoevent/openplay.py — 오픈플레이 이벤트 검출기 (논문 Section 2.4.2-2.4.3)

입력 : SetPieceDetector.run() 결과 DataFrame
       (PossessionDetector 컬럼 + set_piece_type 컬럼 포함)

출력 : 아래 컬럼이 추가된 DataFrame
    - event_name   : 'Pass' | 'Cross' | 'ShotOnTarget' | 'ShotOffTarget'
                     | 'Reception' | 'Interception' | 'Save' | 'Claim'
    - event_player : 이벤트를 실행한 선수 ID
    - event_team   : 선수 팀 ('home' | 'away')
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.AutoEvent.helpers import (
    ball_toward_goal,
    detect_gks,
    get_players,
    gk_in_pa,
    in_attacking_pa,
    in_cross_zone,
    in_shot_zone,
    player_xy,
    shot_on_target,
    team_of,
)
from models.AutoEvent.config import TRACKING_FPS

# 슈팅 판단 시 전방 탐색 윈도우 (5 초 @ 25 fps)
SHOT_WINDOW: int = 5 * TRACKING_FPS

# dead ball 직후 나타나는 set-piece 유형 중 슈팅 이후 발생 가능한 것
# KickOff: 득점 후 재개 → 골도 슈팅으로 분류해야 함
_SHOT_SUCCESSOR_SP: frozenset[str] = frozenset({"CornerKick", "GoalKick", "KickOff"})


class OpenPlayDetector:
    """논문 Section 2.4.2–2.4.3 기준 오픈플레이 이벤트 검출기."""

    def __init__(self, tracking: pd.DataFrame) -> None:
        self.tracking = tracking.copy()
        self.players: list[str] = get_players(self.tracking)
        self.gk_ids: set[str] = detect_gks(self.tracking, self.players)

    def run(self) -> pd.DataFrame:
        return (
            self.add_passing_events()
            .add_receiving_events()
            .tracking
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 패싱 이벤트  (is_loss 프레임)
    # ─────────────────────────────────────────────────────────────────────────

    def add_passing_events(self) -> "OpenPlayDetector":
        """loss 프레임에 Pass / Cross / ShotOnTarget / ShotOffTarget 라벨 부여."""
        # None으로 기록하는 이유: pd.Na 포맷으로 parquet로 저장하면 None으로 타입이 변환되기 때문에 처음부터 None으로 초기화
        self.tracking["event_name"] = None
        self.tracking["event_player"] = None
        self.tracking["event_team"] = None

        n = len(self.tracking)
        for i in range(n):
            row = self.tracking.iloc[i]
            if not row["is_loss"]:
                continue
            if row["ball_state"] != "alive":
                continue
            if row["loss_player"] is None:
                continue
            # Set piece frame은 set_piece_type으로 이미 라벨됨. paper Fig 7a
            # ('open play passing events')는 from_set_piece가 NULL인 케이스만
            # 평가하므로 set piece frame에 별도 Pass 라벨링하지 않음.
            if row.get("set_piece_type") is not None:
                continue

            idx = self.tracking.index[i]
            name, player, team = self._classify_loss(i)
            # name=None ⇒ no recognisable open-play event at this loss
            # frame (paper-faithful: no teammate gain, or out-of-play).
            if name is None:
                continue
            self.tracking.at[idx, "event_name"] = name
            self.tracking.at[idx, "event_player"] = player
            self.tracking.at[idx, "event_team"] = team

        return self

    def _classify_loss(self, i: int) -> tuple[str | None, str, str]:
        """Classify the in-game event at loss frame i.

        Returns (event_name, player, team) where event_name may be None
        when this loss frame doesn't correspond to a recognisable event
        (paper-faithful Pass: must be received by a teammate; otherwise
        the play is an interception / out-of-play / loose ball and we
        don't emit an open-play event).
        """
        row = self.tracking.iloc[i]
        loss_player: str = str(row["loss_player"])
        loss_team: str = str(row["loss_team"])

        # ── 슈팅 판단 ────────────────────────────────────────────────────────
        bx_val = _float(row.get("ball_x"))
        by_val = _float(row.get("ball_y"))
        if (in_shot_zone(bx_val, by_val, loss_team)
                and ball_toward_goal(row, loss_team)
                and self._is_shot_event(i, loss_team)):
            label = ("ShotOnTarget" if shot_on_target(row, loss_team)
                     else "ShotOffTarget")
            return label, loss_player, loss_team

        # Paper Sec 2.4: Pass requires a downstream possession GAIN by a
        # teammate (i.e., the ball is received by the same team). If
        # there is no gain within SHOT_WINDOW frames, or the gain is by
        # an opponent, or by the same player who lost (dribble retain),
        # this is NOT a Pass.
        gain_player, _, gainer_in_pa, gain_i = self._next_gain_info(
            i, loss_team)
        if gain_player is None:
            return None, loss_player, loss_team
        gainer_team = team_of(gain_player)
        if gainer_team != loss_team:
            # Opponent gained the ball — interception, not Pass.
            return None, loss_player, loss_team
        if gain_player == loss_player:
            # Same player regained the ball (brief touch / dribble) —
            # not a real pass.
            return None, loss_player, loss_team

        # ── 크로스 판단 ──────────────────────────────────────────────────────
        # 논문 Sec 2.4.3 — 세 가지 조건 모두 만족해야 cross:
        #  (1) origin in cross zone
        #  (2) next player in control is within active PA
        #  (3) at least one attacking-team player in active PA
        bx = _float(row.get("ball_x"))
        by = _float(row.get("ball_y"))
        if (not np.isnan(bx) and in_cross_zone(bx, by, loss_team)
                and gainer_in_pa
                and self._has_attacker_in_pa(row, loss_team)):
            return "Cross", loss_player, loss_team

        # ── 기본: 패스 (gainer is teammate) ─────────────────────────────────
        return "Pass", loss_player, loss_team

    def _has_attacker_in_pa(self, row: pd.Series, loss_team: str) -> bool:
        """attacking team(loss_team)의 선수 중 active PA 안에 있는 선수가 있는지."""
        for player in self.players:
            if team_of(player) != loss_team:
                continue
            px, py = player_xy(row, player)
            if np.isnan(px) or np.isnan(py):
                continue
            if in_attacking_pa(px, py, loss_team):
                return True
        return False

    def _is_shot_event(self, loss_i: int, loss_team: str) -> bool:
        """loss 프레임 이후 SHOT_WINDOW 내에서 슈팅 결과가 발생했는지.

        조건:
        1. 공이 dead ball → 다음 alive 프레임이 CornerKick / GoalKick / KickOff
        2. 상대 GK 가 PA 안에서 공 탈취(possession gain)
        """
        n = len(self.tracking)
        period = self.tracking.iloc[loss_i].get("period_id")

        for j in range(loss_i + 1, min(loss_i + SHOT_WINDOW + 1, n)):
            row_j = self.tracking.iloc[j]
            if row_j.get("period_id") != period:
                break

            if row_j["ball_state"] == "dead":
                return self._sp_after_dead(j) in _SHOT_SUCCESSOR_SP

            if row_j["is_gain"]:
                gainer = str(row_j["gain_player"])
                if gainer in self.gk_ids:
                    gainer_team = team_of(gainer)
                    if gainer_team != loss_team and gk_in_pa(row_j, gainer):
                        return True
                # 다른 선수가 먼저 잡으면 슈팅 연속이 아님
                break

        return False

    def _sp_after_dead(self, dead_start_i: int) -> str | None:
        """dead_start_i 로 시작하는 dead ball 구간 이후 첫 alive 프레임의 set_piece_type."""
        n = len(self.tracking)
        for k in range(dead_start_i + 1, min(dead_start_i + 3000, n)):
            row_k = self.tracking.iloc[k]
            if row_k["ball_state"] == "alive":
                sp = row_k.get("set_piece_type")
                return sp if sp is not None else None
        return None

    def _next_gain_info(
        self, loss_i: int, loss_team: str
    ) -> tuple[str | None, bool, bool, int]:
        """loss_i 이후 첫 gain 프레임 정보 (dead ball 이전까지).

        Returns
        -------
        (gain_player, is_gk_in_pa, gainer_in_attacking_pa, iloc_of_gain)
        """
        n = len(self.tracking)
        period = self.tracking.iloc[loss_i].get("period_id")

        for j in range(loss_i + 1, min(loss_i + SHOT_WINDOW + 1, n)):
            row_j = self.tracking.iloc[j]
            if row_j.get("period_id") != period:
                break
            if row_j["ball_state"] == "dead":
                break
            if row_j["is_gain"]:
                gainer = str(row_j["gain_player"])
                is_gk_pa = (gainer in self.gk_ids) and gk_in_pa(row_j, gainer)
                gx, gy = player_xy(row_j, gainer)
                in_pa = in_attacking_pa(gx, gy, loss_team)
                return gainer, is_gk_pa, in_pa, j

        return None, False, False, -1

    # ─────────────────────────────────────────────────────────────────────────
    # 리시빙 이벤트  (is_gain 프레임)
    # ─────────────────────────────────────────────────────────────────────────

    def add_receiving_events(self) -> "OpenPlayDetector":
        """gain 프레임에 Reception / Interception / Save / Claim 라벨 부여.

        이미 event_name 이 설정된 프레임(예: 동일 프레임에 loss+gain이 겹치는 경우)은
        건너뜁니다.
        """
        n = len(self.tracking)
        for i in range(n):
            row = self.tracking.iloc[i]
            if not row["is_gain"]:
                continue
            if row["ball_state"] != "alive":
                continue
            if row["gain_player"] is None:
                continue
            # 이미 passing event 가 라벨링 된 프레임은 건너뜀
            if row["event_name"] is not None:
                continue

            idx = self.tracking.index[i]
            name, player, team = self._classify_gain(i)
            if name is not None:
                self.tracking.at[idx, "event_name"] = name
                self.tracking.at[idx, "event_player"] = player
                self.tracking.at[idx, "event_team"] = team

        return self

    def _classify_gain(self, i: int) -> tuple[str | None, str | None, str | None]:
        row = self.tracking.iloc[i]
        gainer: str = str(row["gain_player"])
        gainer_team: str = str(row["gain_team"])

        # ── GK가 PA 안에서 공 획득 → Save / Claim ─────────────────────────
        if gainer in self.gk_ids and gk_in_pa(row, gainer):
            label = "Save" if self._is_preceded_by_shot(i) else "Claim"
            return label, gainer, gainer_team

        # ── 이전 loss 선수로 Reception / Interception 판별 ─────────────────
        prev_loss_player = self._prev_loss_player(i)
        if prev_loss_player is None:
            return "Reception", gainer, gainer_team

        prev_team = team_of(prev_loss_player)
        label = "Reception" if prev_team == gainer_team else "Interception"
        return label, gainer, gainer_team

    def _prev_loss_player(self, gain_i: int) -> str | None:
        """gain_i 이전의 가장 최근 loss_player 반환. dead ball 구간 이전은 무시."""
        period = self.tracking.iloc[gain_i].get("period_id")
        for j in range(gain_i - 1, max(gain_i - SHOT_WINDOW - 1, -1), -1):
            row_j = self.tracking.iloc[j]
            if row_j.get("period_id") != period:
                break
            if row_j["ball_state"] == "dead":
                break
            if row_j["is_loss"] and row_j["loss_player"] is not None:
                return str(row_j["loss_player"])
        return None

    def _is_preceded_by_shot(self, gain_i: int) -> bool:
        """gain_i 직전 SHOT_WINDOW 내에 ShotOnTarget 또는 ShotOffTarget 이 있으면 True."""
        period = self.tracking.iloc[gain_i].get("period_id")
        for j in range(gain_i - 1, max(gain_i - SHOT_WINDOW - 1, -1), -1):
            row_j = self.tracking.iloc[j]
            if row_j.get("period_id") != period:
                break
            if row_j["ball_state"] == "dead":
                break
            en = row_j.get("event_name")
            if en is not None and en in {"ShotOnTarget", "ShotOffTarget"}:
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _float(val) -> float:
    """None / NaN 안전하게 float 변환."""
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        return np.nan