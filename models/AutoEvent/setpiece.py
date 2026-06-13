from __future__ import annotations

import numpy as np
import pandas as pd

from models.AutoEvent.helpers import get_players, player_xy, team_of
from models.AutoEvent.config import (
    CENTER_X,
    CENTER_Y,
    CORNER_TOL,
    EPS_F,
    GOAL_AREA_X,
    GOAL_AREA_X_TOL,
    GOAL_AREA_Y,
    GOAL_AREA_Y_TOL,
    GOAL_POST_Y_MAX,
    GOAL_POST_Y_MIN,
    HALF_TOL,
    ISSUE1_SCAN_FRAMES,
    KICKOFF_X_TOL,
    KICKOFF_Y_TOL,
    LEFT_GOAL_X,
    PENALTY_AREA_X_MAX,
    PENALTY_AREA_Y_MAX,
    PENALTY_AREA_Y_MIN,
    PENALTY_GK_X_TOL,
    PENALTY_GK_Y_TOL,
    PENALTY_MARK_TOL,
    PENALTY_MARK_X,
    PENALTY_OTHER_PLAYERS_TOL,
    PITCH_X,
    PITCH_Y,
    RIGHT_GOAL_X,
    R_PZ,
    THROW_IN_BALL_Z_MIN,
    THROW_IN_TOL,
)


class SetPieceDetector:
    def __init__(self, tracking: pd.DataFrame):
        self.tracking = tracking.copy()
        self.players = get_players(self.tracking)
        self.intervals: list[dict[str, object]] = []

    def run(self):
        return (
            self.add_dead_ball_intervals()
            .add_set_piece_events()
            .tracking
        )

    def add_dead_ball_intervals(self):
        self.tracking["deadball_id"] = None
        self.tracking["deadball_start_frame"] = None
        self.tracking["deadball_end_frame"] = None
        self.tracking["first_inplay_frame"] = None

        self.intervals = []
        states = self.tracking["ball_state"].fillna("dead").astype(str).to_numpy()
        n_frames = len(self.tracking)
        deadball_id = 0
        pos = 0

        while pos < n_frames:
            if states[pos] != "dead":
                pos += 1
                continue

            start_pos = pos
            while pos + 1 < n_frames and states[pos + 1] == "dead":
                pos += 1
            end_pos = pos
            first_inplay_pos = end_pos + 1 if end_pos + 1 < n_frames and states[end_pos + 1] == "alive" else None

            deadball_id += 1
            interval = {
                "deadball_id": deadball_id,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "first_inplay_pos": first_inplay_pos,
            }
            self.intervals.append(interval)

            fill_positions = list(range(start_pos, end_pos + 1))
            if first_inplay_pos is not None:
                fill_positions.append(first_inplay_pos)

            fill_index = self.tracking.index[fill_positions]
            self.tracking.loc[fill_index, "deadball_id"] = deadball_id
            self.tracking.loc[fill_index, "deadball_start_frame"] = self.tracking.iloc[start_pos].get("frame_id", None)
            self.tracking.loc[fill_index, "deadball_end_frame"] = self.tracking.iloc[end_pos].get("frame_id", None)
            if first_inplay_pos is not None:
                self.tracking.loc[fill_index, "first_inplay_frame"] = self.tracking.iloc[first_inplay_pos].get("frame_id", None)

            pos += 1

        # Periods that start directly with alive frames have no preceding dead-ball
        # interval, so their kickoff would be missed. Synthesise a pseudo-interval
        # whose only purpose is to let _kickoff_trigger check that first alive frame.
        covered_first_inplay: set[int] = {
            iv["first_inplay_pos"]
            for iv in self.intervals
            if iv["first_inplay_pos"] is not None
        }
        for period_id in sorted(self.tracking["period_id"].dropna().unique()):
            period_positions = np.where(
                (self.tracking["period_id"] == period_id).to_numpy()
            )[0]
            if len(period_positions) == 0:
                continue
            first_pos = int(period_positions[0])
            if states[first_pos] == "alive" and first_pos not in covered_first_inplay:
                deadball_id += 1
                self.intervals.append(
                    {
                        "deadball_id": deadball_id,
                        "start_pos": None,   # no dead-ball window
                        "end_pos": None,
                        "first_inplay_pos": first_pos,
                    }
                )
                fill_index = self.tracking.index[[first_pos]]
                self.tracking.loc[fill_index, "deadball_id"] = deadball_id
                self.tracking.loc[fill_index, "first_inplay_frame"] = (
                    self.tracking.iloc[first_pos].get("frame_id", None)
                )

        return self

    def add_set_piece_events(self):
        # None으로 기록하는 이유: pd.Na 포맷으로 parquet로 저장하면 None으로 타입이 변환되기 때문에 처음부터 None으로 초기화
        self.tracking["set_piece_type"] = None
        self.tracking["deadball_event"] = None
        self.tracking["trigger_player"] = None
        self.tracking["trigger_team"] = None
        self.tracking["set_piece_source"] = None
        self.tracking["set_piece_pattern"] = None

        for interval in self.intervals:
            first_inplay_pos = interval["first_inplay_pos"]
            if first_inplay_pos is None:
                continue

            classification = self._classify_interval(interval)
            if classification is None:
                continue

            first_index = self.tracking.index[first_inplay_pos]
            self.tracking.at[first_index, "set_piece_type"] = classification["set_piece_type"]
            self.tracking.at[first_index, "trigger_player"] = classification.get("trigger_player", None)
            self.tracking.at[first_index, "trigger_team"] = classification.get("trigger_team", None)
            self.tracking.at[first_index, "set_piece_source"] = classification.get("source", None)
            self.tracking.at[first_index, "set_piece_pattern"] = classification.get("pattern", None)

        return self

    def _classify_interval(self, interval: dict[str, object]) -> dict[str, object] | None:
        first_inplay_pos = int(interval["first_inplay_pos"])
        first_row = self.tracking.iloc[first_inplay_pos]

        # Issue 1: first player in control이 εf 이상 이동한 프레임(execution frame)을 pattern 확인에 사용.
        execution_row = self._find_execution_frame(first_inplay_pos)

        # Pre-compute all triggers.
        kickoff_cands       = self._kickoff_trigger(interval)
        penalty_cands       = self._penalty_trigger(interval)
        corner_cands        = self._corner_trigger(interval)
        throw_in_cands      = self._throw_in_trigger(interval)
        goal_kick_cands     = self._goal_kick_trigger(interval)
        corner_incomplete   = self._corner_incomplete_trigger(interval)
        throw_in_incomplete = self._throw_in_incomplete_trigger(interval)
        throw_in_ball_side  = self._throw_in_ball_side(interval)

        # === 1차: trigger + pattern ===

        if kickoff_cands:
            player = self._resolve_kickoff_pattern(execution_row, kickoff_cands)
            if player is not None:
                return self._build_result("KickOff", player, "kickoff_trigger", "kickoff_pattern")
            # Pattern 실패: trigger 신뢰 → trigger-only KickOff (FreeKick 낙오 방지)
            player = next(iter(kickoff_cands))
            return self._build_result("KickOff", player, "kickoff_trigger_only", "kickoff_no_pattern")

        elif penalty_cands:
            player = self._resolve_penalty_pattern(execution_row, penalty_cands)
            if player is not None:
                return self._build_result("Penalty", player, "penalty_trigger", "penalty_pattern")
            player = self._resolve_penalty_trigger_only(execution_row, penalty_cands)
            if player is not None:
                return self._build_result("Penalty", player, "penalty_trigger_only", "penalty_no_pattern")

        elif corner_cands:
            corner_trigger_player = self._select_corner_trigger_player(interval, corner_cands)
            player = self._resolve_corner_pattern(execution_row, corner_cands)
            if player is not None:
                return self._build_result("CornerKick", player, "corner_trigger", "corner_pattern")
            player = self._resolve_throw_in_pattern(execution_row, throw_in_cands, throw_in_ball_side)
            if player is not None:
                return self._build_result("ThrowIn", player, "corner_to_throwin", "throwin_pattern")
            player = self._resolve_goal_kick_pattern(execution_row, goal_kick_cands)
            if player is not None:
                return self._build_result("GoalKick", player, "corner_to_goalkick", "goalkick_pattern")
            player = self._resolve_corner_trigger_only(first_row, execution_row, corner_trigger_player)
            if player is not None:
                return self._build_result("CornerKick", player, "corner_trigger_only", "corner_no_pattern")

        else:
            # ball_z high + ball-near-sideline → strong ThrowIn signal.
            # (J03WMX data: 100% of true throw-ins have ball_y within 2m of sideline
            #  at first inplay frame; free/goal/kickoff are far from sideline.)
            ball_z_first = float(first_row.get("ball_z", 0) or 0)
            ball_y_first = float(first_row.get("ball_y", PITCH_Y / 2) or 0)
            sideline_dist = min(abs(ball_y_first), abs(PITCH_Y - ball_y_first))
            is_thrown_by_z = (ball_z_first > THROW_IN_BALL_Z_MIN
                              and sideline_dist < 2.0)

            player = self._resolve_throw_in_pattern(execution_row, throw_in_cands, throw_in_ball_side)
            if player is not None:
                return self._build_result("ThrowIn", player, "throwin_trigger", "throwin_pattern")

            # ball_z + sideline rule → ThrowIn even when sideline-player trigger missed
            if is_thrown_by_z:
                player = (
                    next(iter(throw_in_cands)) if throw_in_cands
                    else next(iter(throw_in_incomplete)) if throw_in_incomplete
                    else self._find_closest_player_to_ball(
                        execution_row if execution_row is not None else first_row
                    )
                )
                if player is not None:
                    return self._build_result(
                        "ThrowIn", player, "ball_z_high", "throwin_z"
                    )

            player = self._resolve_goal_kick_pattern(execution_row, goal_kick_cands)
            if player is not None:
                return self._build_result("GoalKick", player, "goalkick_trigger", "goalkick_pattern")

            # FreeKick: sideline/goal area 근처 선수는 자동 제외 → ThrowIn/GoalKick 혼동 방지
            result = self._resolve_free_kick(execution_row, "free_kick_pattern")
            if result is not None:
                return result

            # Issue 2: incomplete triggers (Ci* → Ti*)
            if corner_incomplete:
                player = next(iter(corner_incomplete))
                return self._build_result("CornerKick", player, "corner_incomplete", "corner_incomplete")
            if throw_in_incomplete:
                player = next(iter(throw_in_incomplete))
                return self._build_result("ThrowIn", player, "throwin_incomplete", "throwin_incomplete")

            # KickOff pattern-only 복구: trigger 없이 execution_row에서 중앙+점유 확인
            player = self._resolve_kickoff_pattern(execution_row, set(self.players))
            if player is not None:
                return self._build_result("KickOff", player, "kickoff_pattern_only", "kickoff_pattern")

        # === Issue 3: 모든 trigger/pattern 실패 ===
        return self._classify_trigger_only(
            first_row,
            execution_row,
            kickoff_cands, penalty_cands,
            corner_cands, throw_in_cands,
            corner_incomplete, throw_in_incomplete,
            goal_kick_cands,
        )

    def _find_closest_player_to_ball(self, row: pd.Series) -> str | None:
        """row 시점에서 공과 가장 가까운 선수 ID 반환."""
        bx = row.get("ball_x", np.nan)
        by = row.get("ball_y", np.nan)
        if pd.isna(bx) or pd.isna(by):
            return None
        bx, by = float(bx), float(by)
        best_player: str | None = None
        best_d2 = float("inf")
        for player in self.players:
            px, py = player_xy(row, player)
            if np.isnan(px) or np.isnan(py):
                continue
            d2 = (px - bx) ** 2 + (py - by) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_player = player
        return best_player

    def _find_execution_frame(self, first_inplay_pos: int) -> pd.Series:
        """Issue 1: first_inplay_pos부터 스캔하여 첫 possession player(controller_id)가
        εf(EPS_F) 이상 이동한 프레임을 반환. 찾지 못하면 first_inplay_pos 프레임을 반환.

        논문 Algorithm S1 Issue 1:
          'the first player in control moves more than εf'
          → 해당 프레임을 execution frame으로 사용하여 pattern 확인에 쓴다.
        """
        n = len(self.tracking)
        end_scan = min(first_inplay_pos + ISSUE1_SCAN_FRAMES, n)

        first_controller: str | None = None
        first_controller_pos: tuple[float, float] | None = None

        for pos in range(first_inplay_pos, end_scan):
            row = self.tracking.iloc[pos]
            controller = row.get("controller_id", None)
            if controller is None:
                continue

            px, py = player_xy(row, controller)
            if np.isnan(px) or np.isnan(py):
                continue

            if first_controller is None:
                # 첫 possession player와 그 초기 위치 기록
                first_controller = controller
                first_controller_pos = (px, py)
                continue

            if controller != first_controller:
                # 다른 선수가 공을 가졌으면 더 이상 추적할 필요 없음
                break

            dx = px - first_controller_pos[0]
            dy = py - first_controller_pos[1]
            if float(np.hypot(dx, dy)) >= EPS_F:
                return row  # execution frame 발견

        # 탐색 실패 → first_inplay_pos 기본값 반환
        return self.tracking.iloc[first_inplay_pos]

    def _classify_trigger_only(
        self,
        first_row: pd.Series,
        execution_row: pd.Series,
        kickoff_cands: set[str],
        penalty_cands: set[str],
        corner_cands: set[str],
        throw_in_cands: set[str],
        corner_incomplete: set[str],
        throw_in_incomplete: set[str],
        goal_kick_cands: set[str],
    ) -> dict[str, object] | None:
        """Issue 3: K → P → C(complete) → T(complete) → C*(incomplete) → T*(incomplete) → G → FreeKick?"""
        if kickoff_cands:
            player = next(iter(kickoff_cands))
            return self._build_result("KickOff", player, "issue3_trigger_only", "kickoff_trigger_only")
        if penalty_cands:
            player = self._resolve_penalty_trigger_only(execution_row, penalty_cands)
            if player is not None:
                return self._build_result("Penalty", player, "issue3_trigger_only", "penalty_trigger_only")
        if corner_cands:
            player = self._resolve_corner_trigger_only(
                first_row,
                execution_row,
                self._select_corner_trigger_player_from_candidates(execution_row, corner_cands),
            )
            if player is not None:
                return self._build_result("CornerKick", player, "issue3_trigger_only", "corner_trigger_only")
        if throw_in_cands:
            player = next(iter(throw_in_cands))
            return self._build_result("ThrowIn", player, "issue3_trigger_only", "throwin_trigger_only")
        if corner_incomplete:
            player = next(iter(corner_incomplete))
            return self._build_result("CornerKick", player, "issue3_incomplete_trigger", "corner_incomplete")
        if throw_in_incomplete:
            player = next(iter(throw_in_incomplete))
            return self._build_result("ThrowIn", player, "issue3_incomplete_trigger", "throwin_incomplete")
        if goal_kick_cands:
            player = next(iter(goal_kick_cands))
            return self._build_result("GoalKick", player, "issue3_trigger_only", "goalkick_trigger_only")
        # 모든 trigger 실패 → FreeKick? (confused)
        return self._build_result("FreeKick?", "unknown", "issue3_no_trigger", "no_trigger")

    def _build_result(self, set_piece_type: str, trigger_player: str, source: str, pattern: str) -> dict[str, object]:
        return {
            "set_piece_type": set_piece_type,
            "trigger_player": trigger_player,
            "trigger_team": team_of(trigger_player) if trigger_player != "unknown" else None,
            "source": source,
            "pattern": pattern,
        }

    def _resolve_free_kick(self, first_row: pd.Series, source: str) -> dict[str, object] | None:
        candidates = self._players_in_possession_zone(first_row, self.players)
        if not candidates:
            return None
        # sideline 근처 선수 제외(ThrowIn 영역) / goal area 선수 제외(GoalKick 영역)
        filtered: list[tuple[str, float]] = []
        for player, dist in candidates:
            px, py = player_xy(first_row, player)
            if np.isnan(py):
                continue
            if py <= THROW_IN_TOL or py >= (PITCH_Y - THROW_IN_TOL):
                continue  # sideline band → ThrowIn territory
            if self._in_own_goal_area(px, py, team_of(player)):
                continue  # goal area → GoalKick territory
            filtered.append((player, dist))
        if not filtered:
            return None
        trigger_player = min(filtered, key=lambda item: item[1])[0]
        return self._build_result("FreeKick", trigger_player, source, "free_kick_pattern")

    def _last_visible_ball_row_before(self, start_pos: int) -> pd.Series | None:
        for pos in range(start_pos - 1, -1, -1):
            row = self.tracking.iloc[pos]
            ball_y = float(row.get("ball_y", np.nan))
            if not np.isnan(ball_y):
                return row
        return None

    def _throw_in_ball_side(self, interval: dict[str, object]) -> str | None:
        start_pos = interval.get("start_pos")
        if start_pos is None:
            return None
        ball_row = self._last_visible_ball_row_before(int(start_pos))
        if ball_row is None:
            return None
        return self._sideline_side(float(ball_row.get("ball_y", np.nan)))

    def _sideline_side(self, y: float) -> str | None:
        if np.isnan(y):
            return None
        if y <= THROW_IN_TOL:
            return "bottom"
        if y >= (PITCH_Y - THROW_IN_TOL):
            return "top"
        return None

    def _matches_sideline_side(self, y: float, side: str | None) -> bool:
        return side is not None and self._sideline_side(y) == side

    def _kickoff_trigger(self, interval: dict[str, object]) -> set[str]:
        # Kickoff formation is only reliable on the first in-play frame:
        # - Period-start intervals have no dead-ball window at all.
        # - Halftime dead-ball windows show players in the dressing room.
        first_inplay_pos = interval.get("first_inplay_pos")
        if first_inplay_pos is None:
            return set()
        row = self.tracking.iloc[int(first_inplay_pos)]

        # Ball-at-center is a strong kickoff signal: bypass own-halves check when
        # ball is on the center mark at first inplay (handles post-goal scenarios
        # where players are still scattered from celebration).
        bx = row.get("ball_x")
        by = row.get("ball_y")
        ball_at_center = (
            pd.notna(bx) and pd.notna(by)
            and abs(float(bx) - CENTER_X) <= KICKOFF_X_TOL
            and abs(float(by) - CENTER_Y) <= KICKOFF_Y_TOL
        )

        if not ball_at_center and not self._players_in_own_halves(row):
            return set()

        candidates: set[str] = set()
        for player in self.players:
            px, py = player_xy(row, player)
            if np.isnan(px) or np.isnan(py):
                continue
            if abs(px - CENTER_X) <= KICKOFF_X_TOL and abs(py - CENTER_Y) <= KICKOFF_Y_TOL:
                candidates.add(player)
        return candidates

    def _penalty_trigger(self, interval: dict[str, object]) -> set[str]:
        candidates: set[str] = set()
        if interval["start_pos"] is None:
            return candidates
        for pos in range(int(interval["start_pos"]), int(interval["end_pos"]) + 1):
            row = self.tracking.iloc[pos]
            for attacking_team in ("home", "away"):
                kicker = self._penalty_kicker_for_team(row, attacking_team)
                if kicker is not None:
                    candidates.add(kicker)
        return candidates

    def _corner_trigger(self, interval: dict[str, object]) -> set[str]:
        candidates: set[str] = set()
        if interval["start_pos"] is None:
            return candidates
        for pos in range(int(interval["start_pos"]), int(interval["end_pos"]) + 1):
            row = self.tracking.iloc[pos]
            for player in self.players:
                px, py = player_xy(row, player)
                if self._in_active_corner_zone(px, py, team_of(player)):
                    candidates.add(player)
        return candidates

    def _throw_in_trigger(self, interval: dict[str, object]) -> set[str]:
        candidates: set[str] = set()
        if interval["start_pos"] is None:
            return candidates
        for pos in range(int(interval["start_pos"]), int(interval["end_pos"]) + 1):
            row = self.tracking.iloc[pos]
            for player in self.players:
                _, py = player_xy(row, player)
                if np.isnan(py):
                    continue
                if py <= THROW_IN_TOL or py >= (PITCH_Y - THROW_IN_TOL):
                    candidates.add(player)
        return candidates

    def _goal_kick_trigger(self, interval: dict[str, object]) -> set[str]:
        candidates: set[str] = set()
        if interval["start_pos"] is None:
            return candidates
        for pos in range(int(interval["start_pos"]), int(interval["end_pos"]) + 1):
            row = self.tracking.iloc[pos]
            for player in self.players:
                px, py = player_xy(row, player)
                if self._in_own_goal_area(px, py, team_of(player)):
                    candidates.add(player)
        return candidates

    def _corner_incomplete_trigger(self, interval: dict[str, object]) -> set[str]:
        """dead-ball 구간 중 일부 구간만 corner zone에 있었고, 이후 사라진 선수 (Issue 2/3)."""
        if interval["start_pos"] is None:
            return set()
        start_pos = int(interval["start_pos"])
        end_pos = int(interval["end_pos"])
        ever_in_corner: dict[str, int] = {}   # player → 마지막으로 corner zone에 있었던 pos
        for pos in range(start_pos, end_pos + 1):
            row = self.tracking.iloc[pos]
            for player in self.players:
                px, py = player_xy(row, player)
                if self._in_active_corner_zone(px, py, team_of(player)):
                    ever_in_corner[player] = pos
        # incomplete: corner zone에 있었지만 dc 이전에 사라진 선수
        incomplete: set[str] = set()
        for player, last_pos in ever_in_corner.items():
            if last_pos < end_pos:
                # [last_pos+1, end_pos] 에서 해당 선수가 트래킹 안 됨
                missing = all(
                    np.isnan(player_xy(self.tracking.iloc[p], player)[0])
                    for p in range(last_pos + 1, end_pos + 1)
                )
                if missing:
                    incomplete.add(player)
        return incomplete

    def _throw_in_incomplete_trigger(self, interval: dict[str, object]) -> set[str]:
        """dead-ball 구간 중 일부 구간만 sideline 근처에 있었고, 이후 사라진 선수 (Issue 2/3)."""
        if interval["start_pos"] is None:
            return set()
        start_pos = int(interval["start_pos"])
        end_pos = int(interval["end_pos"])
        ever_near_line: dict[str, int] = {}
        for pos in range(start_pos, end_pos + 1):
            row = self.tracking.iloc[pos]
            for player in self.players:
                _, py = player_xy(row, player)
                if np.isnan(py):
                    continue
                if py <= THROW_IN_TOL or py >= (PITCH_Y - THROW_IN_TOL):
                    ever_near_line[player] = pos
        incomplete: set[str] = set()
        for player, last_pos in ever_near_line.items():
            if last_pos < end_pos:
                missing = all(
                    np.isnan(player_xy(self.tracking.iloc[p], player)[0])
                    for p in range(last_pos + 1, end_pos + 1)
                )
                if missing:
                    incomplete.add(player)
        return incomplete

    def _resolve_kickoff_pattern(self, first_row: pd.Series, candidates: set[str]) -> str | None:
        return self._resolve_pattern_player(
            first_row,
            candidates,
            lambda player, px, py: abs(px - CENTER_X) <= KICKOFF_X_TOL and abs(py - CENTER_Y) <= KICKOFF_Y_TOL,
        )

    def _resolve_penalty_pattern(self, first_row: pd.Series, candidates: set[str]) -> str | None:
        return self._resolve_pattern_player(
            first_row,
            candidates,
            lambda player, px, py: self._near_active_penalty_mark(px, py, team_of(player)),
        )

    def _resolve_penalty_trigger_only(self, first_row: pd.Series, candidates: set[str]) -> str | None:
        valid_candidates = [
            player
            for player in candidates
            if self._player_near_penalty_mark(first_row, player)
        ]
        if not valid_candidates:
            return None

        players_in_zone = self._players_in_possession_zone(first_row, valid_candidates)
        if len(players_in_zone) == 1:
            return players_in_zone[0][0]
        if len(players_in_zone) > 1:
            return min(players_in_zone, key=lambda item: item[1])[0]

        if len(valid_candidates) == 1:
            return valid_candidates[0]
        return None

    def _resolve_corner_trigger_only(
        self,
        first_row: pd.Series,
        execution_row: pd.Series,
        trigger_player: str | None,
    ) -> str | None:
        if trigger_player is None:
            return None
        if self._player_in_active_corner_zone(first_row, trigger_player):
            return trigger_player
        if self._player_in_active_corner_zone(execution_row, trigger_player):
            return trigger_player
        return None

    def _resolve_corner_pattern(self, first_row: pd.Series, candidates: set[str]) -> str | None:
        return self._resolve_pattern_player(
            first_row,
            candidates,
            lambda player, px, py: self._in_active_corner_zone(px, py, team_of(player)),
        )

    def _resolve_throw_in_pattern(self, first_row: pd.Series, candidates: set[str], ball_side: str | None) -> str | None:
        if ball_side is None:
            return None
        return self._resolve_pattern_player(
            first_row,
            candidates,
            lambda _player, _px, py: self._matches_sideline_side(py, ball_side),
        )

    def _resolve_goal_kick_pattern(self, first_row: pd.Series, candidates: set[str]) -> str | None:
        return self._resolve_pattern_player(
            first_row,
            candidates,
            lambda player, px, py: self._in_own_goal_area(px, py, team_of(player)),
        )

    def _select_corner_trigger_player(self, interval: dict[str, object], candidates: set[str]) -> str | None:
        if not candidates or interval["start_pos"] is None:
            return None
        start_pos = int(interval["start_pos"])
        end_pos = int(interval["end_pos"])
        for pos in range(end_pos, start_pos - 1, -1):
            row = self.tracking.iloc[pos]
            player = self._select_corner_trigger_player_from_candidates(row, candidates)
            if player is not None:
                return player
        return None

    def _select_corner_trigger_player_from_candidates(
        self,
        row: pd.Series,
        candidates: set[str],
    ) -> str | None:
        valid_candidates: list[tuple[str, float]] = []
        for player in candidates:
            px, py = player_xy(row, player)
            if not self._in_active_corner_zone(px, py, team_of(player)):
                continue
            valid_candidates.append((player, self._corner_zone_distance(px, py, team_of(player))))
        if not valid_candidates:
            return None
        return min(valid_candidates, key=lambda item: item[1])[0]

    def _resolve_pattern_player(
        self,
        first_row: pd.Series,
        candidates: set[str],
        zone_check,
    ) -> str | None:
        if not candidates:
            return None

        valid_candidates: list[tuple[str, float]] = []
        for player, dist in self._players_in_possession_zone(first_row, list(candidates)):
            px, py = player_xy(first_row, player)
            if zone_check(player, px, py):
                valid_candidates.append((player, dist))

        if not valid_candidates:
            return None

        return min(valid_candidates, key=lambda item: item[1])[0]

    def _players_in_possession_zone(self, row: pd.Series, candidates: list[str]) -> list[tuple[str, float]]:
        ball_x = float(row.get("ball_x", np.nan))
        ball_y = float(row.get("ball_y", np.nan))
        if np.isnan(ball_x) or np.isnan(ball_y):
            return []

        players_in_zone: list[tuple[str, float]] = []
        for player in candidates:
            px, py = player_xy(row, player)
            if np.isnan(px) or np.isnan(py):
                continue
            dist = float(np.hypot(px - ball_x, py - ball_y))
            if dist <= R_PZ:
                players_in_zone.append((player, dist))
        return players_in_zone

    def _players_in_own_halves(self, row: pd.Series) -> bool:
        for player in self.players:
            px, _ = player_xy(row, player)
            if np.isnan(px):
                continue
            if player.startswith("home_") and px > CENTER_X + HALF_TOL:
                return False
            if player.startswith("away_") and px < CENTER_X - HALF_TOL:
                return False
        return True

    def _penalty_kicker_for_team(self, row: pd.Series, attacking_team: str) -> str | None:
        defending_team = "away" if attacking_team == "home" else "home"
        goal_x = RIGHT_GOAL_X if attacking_team == "home" else LEFT_GOAL_X
        mark_x = PITCH_X - PENALTY_MARK_X if attacking_team == "home" else PENALTY_MARK_X

        goal_line_players = []
        kicker_candidates = []

        for player in self.players:
            px, py = player_xy(row, player)
            if np.isnan(px) or np.isnan(py):
                continue

            team = team_of(player)
            if team == defending_team and self._in_goal_line_box(px, py, goal_x):
                goal_line_players.append(player)

            if team == attacking_team and self._in_penalty_mark_box(px, py, mark_x):
                kicker_candidates.append(player)

        if len(goal_line_players) != 1 or len(kicker_candidates) != 1:
            return None

        kicker = kicker_candidates[0]
        for player in self.players:
            if player in {goal_line_players[0], kicker}:
                continue

            px, py = player_xy(row, player)
            if np.isnan(px) or np.isnan(py):
                continue

            if self._in_active_penalty_area(px, py, attacking_team):
                return None

            radius = float(np.hypot(px - mark_x, py - CENTER_Y))
            if radius < 9.15 - PENALTY_OTHER_PLAYERS_TOL:
                return None

        return kicker

    def _in_goal_line_box(self, px: float, py: float, goal_x: float) -> bool:
        return (
            abs(px - goal_x) <= PENALTY_GK_X_TOL
            and (GOAL_POST_Y_MIN - PENALTY_GK_Y_TOL) <= py <= (GOAL_POST_Y_MAX + PENALTY_GK_Y_TOL)
        )

    def _in_penalty_mark_box(self, px: float, py: float, mark_x: float) -> bool:
        return abs(px - mark_x) <= PENALTY_MARK_TOL and abs(py - CENTER_Y) <= PENALTY_MARK_TOL

    def _player_near_penalty_mark(self, row: pd.Series, player: str) -> bool:
        px, py = player_xy(row, player)
        if np.isnan(px) or np.isnan(py):
            return False
        return self._near_active_penalty_mark(px, py, team_of(player))

    def _player_in_active_corner_zone(self, row: pd.Series, player: str) -> bool:
        px, py = player_xy(row, player)
        return self._in_active_corner_zone(px, py, team_of(player))

    def _near_active_penalty_mark(self, px: float, py: float, attacking_team: str) -> bool:
        mark_x = PITCH_X - PENALTY_MARK_X if attacking_team == "home" else PENALTY_MARK_X
        return self._in_penalty_mark_box(px, py, mark_x)

    def _in_active_penalty_area(self, px: float, py: float, attacking_team: str) -> bool:
        if np.isnan(px) or np.isnan(py):
            return False
        if not (PENALTY_AREA_Y_MIN + PENALTY_OTHER_PLAYERS_TOL <= py <= PENALTY_AREA_Y_MAX - PENALTY_OTHER_PLAYERS_TOL):
            return False
        if attacking_team == "home":
            return px >= (PITCH_X - PENALTY_AREA_X_MAX + PENALTY_OTHER_PLAYERS_TOL)
        return px <= PENALTY_AREA_X_MAX - PENALTY_OTHER_PLAYERS_TOL

    def _in_active_corner_zone(self, px: float, py: float, attacking_team: str) -> bool:
        if np.isnan(px) or np.isnan(py):
            return False
        corner_x = RIGHT_GOAL_X if attacking_team == "home" else LEFT_GOAL_X
        return (
            float(np.hypot(px - corner_x, py - 0.0)) <= CORNER_TOL
            or float(np.hypot(px - corner_x, py - PITCH_Y)) <= CORNER_TOL
        )

    def _corner_zone_distance(self, px: float, py: float, attacking_team: str) -> float:
        corner_x = RIGHT_GOAL_X if attacking_team == "home" else LEFT_GOAL_X
        return min(
            float(np.hypot(px - corner_x, py - 0.0)),
            float(np.hypot(px - corner_x, py - PITCH_Y)),
        )

    def _in_own_goal_area(self, px: float, py: float, team: str) -> bool:
        if np.isnan(px) or np.isnan(py):
            return False
        x_min, x_max = self._goal_area_x_bounds(team)
        y_min = CENTER_Y - GOAL_AREA_Y - GOAL_AREA_Y_TOL
        y_max = CENTER_Y + GOAL_AREA_Y + GOAL_AREA_Y_TOL
        return x_min <= px <= x_max and y_min <= py <= y_max

    def _goal_area_x_bounds(self, team: str) -> tuple[float, float]:
        if team == "home":
            return -GOAL_AREA_X_TOL, GOAL_AREA_X + GOAL_AREA_X_TOL
        return PITCH_X - GOAL_AREA_X - GOAL_AREA_X_TOL, PITCH_X + GOAL_AREA_X_TOL