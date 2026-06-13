"""Build raw_gt_synced/<MID>.parquet directly from DFL raw XML.

1. (extract_match): parse DFL raw XML into a GT frame, aligning
    EventTime to tracking frame_id.
2. (build_synced): attach sync_ts / sync_quality / sync_drift by
    matching each GT row to the nearest same-player elastic event.

"""
from __future__ import annotations
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dateutil.parser import isoparse

from config import (
    TRACKING_FPS,
    PERIOD_START_FRAME,
    DFL_TO_SPADL,
    KIND_MATCH_WINDOW_S,
    ANY_MATCH_WINDOW_S
)

# 1. extract_match
def build_player_map(meta_root) -> dict[str, str]:
    """Map DFL-OBJ-XXX -> 'home_7' / 'away_2' using ShirtNumber and team Role."""
    mapping: dict[str, str] = {}
    for team in meta_root.iter("Team"):
        role = team.attrib.get("Role")  # 'home' or 'guest'
        side = "home" if role == "home" else "away"
        for p in team.iter("Player"):
            pid = p.attrib.get("PersonId")
            shirt = p.attrib.get("ShirtNumber")
            if pid and shirt is not None:
                mapping[pid] = f"{side}_{shirt}"
    return mapping

def find_kickoffs(events_root) -> dict[int, datetime]:
    """Locate first/second-half kickoff EventTimes."""
    kickoffs: dict[int, datetime] = {}
    for ev in events_root.iter("Event"):
        for ko in ev.iter("KickOff"):
            gs = ko.attrib.get("GameSection")
            if gs == "firstHalf" and 1 not in kickoffs:
                kickoffs[1] = isoparse(ev.attrib["EventTime"])
            elif gs == "secondHalf" and 2 not in kickoffs:
                kickoffs[2] = isoparse(ev.attrib["EventTime"])
    return kickoffs

def event_to_frame(event_time: datetime, period_id: int,
                   kickoffs: dict[int, datetime]) -> int:
    """Convert real-world EventTime to tracking frame_id."""
    elapsed = (event_time - kickoffs[period_id]).total_seconds()
    return PERIOD_START_FRAME[period_id] + round(elapsed * TRACKING_FPS)

def primary_event_kind(event_el) -> tuple[Optional[str], dict]:
    """Determine the primary event type from an <Event> element's children.

    Returns (event_type, attrib_payload) — event_type is the DFL action tag
    (e.g. 'Pass', 'OtherBallAction', 'TacklingGame', 'KickOff', etc.),
    and attrib_payload is a flattened dict of relevant child attributes.
    """
    for child in event_el:
        tag = child.tag

        # Set piece wrappers: KickOff, ThrowIn, GoalKick, CornerKick,
        # FreeKick, Penalty, RefereeBall, FinalWhistle, etc.
        if tag in {
            "KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick",
            "Penalty", "RefereeBall", "FinalWhistle",
        }:
            payload = dict(child.attrib)
            for play in child.iter("Play"):
                payload.update({
                    "play_player": play.attrib.get("Player"),
                    "play_team":   play.attrib.get("Team"),
                    "play_eval":   play.attrib.get("Evaluation"),
                })
            if tag == "Penalty" and "play_player" not in payload:
                for shot in child.iter("ShotAtGoal"):
                    payload.update({
                        "play_player": shot.attrib.get("Player"),
                        "play_team":   shot.attrib.get("Team"),
                        "play_eval":   shot.attrib.get("Evaluation"),
                        "shot_kind":   shot.attrib.get("Kind", "ShotAtGoal"),
                    })
                    break
            return tag, payload

        # Open-play actions
        if tag == "Play":
            payload = dict(child.attrib)
            for sub in child:
                if sub.tag == "Pass":
                    return "Pass", payload
                if sub.tag == "Cross":
                    return "Cross", payload
                if sub.tag == "ShotAtGoal":
                    payload["shot_kind"] = sub.attrib.get("Kind", "ShotAtGoal")
                    for sub2 in sub:
                        payload["shot_outcome"] = sub2.tag
                    return "ShotAtGoal", payload
            return "Play_other", payload

        if tag == "ShotAtGoal":
            payload = dict(child.attrib)
            for outcome in child:
                payload["shot_outcome"] = outcome.tag
                payload.update({f"out_{k}": v for k, v in outcome.attrib.items()})
                break
            return "ShotAtGoal", payload

        if tag in {"OtherBallAction", "TacklingGame", "BallClaiming",
                   "Foul", "Offside", "Substitution", "Caution",
                   "PenaltyNotAwarded", "ChanceWithoutShot", "Run",
                   "Delete", "VideoAssistantAction", "PlayerNotSentOff"}:
            return tag, dict(child.attrib)

    return None, {}


def extract_match(match_id: str, raw_dir: Path) -> pd.DataFrame:
    """Build a clean GT DataFrame for one match from its DFL raw XML."""
    match_dir = Path(raw_dir) / match_id
    # Support both naming conventions: DFL_02_01_matchinformation*.xml
    # and DFL-02.01-Spielinformationen*.xml
    meta_candidates = (list(match_dir.glob("DFL_02_01_matchinformation*.xml"))
                       + list(match_dir.glob("DFL-02.01-Spielinformationen*.xml")))
    events_candidates = (list(match_dir.glob("DFL_03_02_events_raw*.xml"))
                         + list(match_dir.glob("DFL-03.02-Ereignisdaten-Spiel-Roh*.xml")))
    if not meta_candidates or not events_candidates:
        raise FileNotFoundError(f"Missing meta/events for {match_dir}")
    meta_xml = meta_candidates[0]
    events_xml = events_candidates[0]

    meta_root = ET.parse(meta_xml).getroot()
    events_root = ET.parse(events_xml).getroot()

    info = meta_root.find(".//MatchInformation/General")
    if info is None:
        raise RuntimeError("MatchInformation/General missing")
    kickoff1 = isoparse(info.attrib["KickoffTime"])
    home_id = info.attrib["HomeTeamId"]
    guest_id = info.attrib["GuestTeamId"]

    kickoffs = find_kickoffs(events_root)
    if 1 not in kickoffs:
        kickoffs[1] = kickoff1  # fallback
    if 2 not in kickoffs:
        raise RuntimeError(f"secondHalf KickOff missing for {match_id}")

    player_map = build_player_map(meta_root)

    rows = []
    for ev in events_root.iter("Event"):
        et_str = ev.attrib.get("EventTime")
        if not et_str:
            continue
        et = isoparse(et_str)
        period_id = 1 if et < kickoffs[2] else 2
        try:
            frame_id = event_to_frame(et, period_id, kickoffs)
        except KeyError:
            continue

        kind, payload = primary_event_kind(ev)
        if kind is None:
            continue

        person_id = (
            payload.get("Player")
            or payload.get("play_player")
            or payload.get("Winner")  # TacklingGame primary actor
            or payload.get("Loser")   # fallback
        )
        player = player_map.get(person_id) if person_id else None

        team_dfl = payload.get("Team") or payload.get("WinnerTeam") or payload.get("play_team")
        team = None
        if team_dfl == home_id:
            team = "home"
        elif team_dfl == guest_id:
            team = "away"

        rows.append({
            "frame_id": frame_id,
            "period_id": period_id,
            "event_time": et_str,
            "event_kind": kind,
            "player_id": player,
            "person_id": person_id,
            "team": team,
            "x": float(ev.attrib.get("X-Position", "nan")),
            "y": float(ev.attrib.get("Y-Position", "nan")),
            "evaluation": payload.get("Evaluation") or payload.get("play_eval"),
            "winner_id": payload.get("Winner"),
            "loser_id": payload.get("Loser"),
            "winner_role": payload.get("WinnerRole"),
            "loser_role": payload.get("LoserRole"),
            "winner_team": payload.get("WinnerTeam"),
            "loser_team": payload.get("LoserTeam"),
            "possession_change": payload.get("PossessionChange"),
            "winner_result": payload.get("WinnerResult"),
            "shot_kind": payload.get("shot_kind"),
            "shot_outcome": payload.get("shot_outcome"),
            "defensive_clearance": payload.get("DefensiveClearance"),
            "sub_kind": payload.get("sub_kind"),
            "event_id": ev.attrib.get("EventId"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["period_id", "frame_id"]).reset_index(drop=True)
    df["timestamp"] = (df["frame_id"] - df["period_id"].map(PERIOD_START_FRAME)) / TRACKING_FPS
    return df

# 2. build_synced
def _ts_to_seconds(t) -> float:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return np.nan
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t)
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)

def build_synced(raw: pd.DataFrame, match_id: str, data_path: str) -> pd.DataFrame:
    """Attach sync_ts / sync_quality / sync_drift to an extracted GT frame.

    ``raw`` is the in-memory output of ``extract_match`` (no longer read
    back from a raw_gt/<MID>.parquet intermediate).
    """
    raw = raw.reset_index(drop=True)
    elastic_path = Path(data_path) / match_id / "event.parquet"
    if not elastic_path.exists():
        raise FileNotFoundError(f"Missing elastic event file: {elastic_path}")
    el = pd.read_parquet(elastic_path).copy()
    el["ts_s"] = el["timestamp"].apply(_ts_to_seconds)

    out = raw.copy()
    out["sync_ts"] = np.nan
    out["sync_quality"] = pd.Series([None] * len(out), dtype=object)

    # Group elastic by (period, player) once for fast lookup.
    el_by_pp: dict[tuple[int, str], pd.DataFrame] = {
        key: grp.sort_values("ts_s").reset_index(drop=True)
        for key, grp in el.groupby(["period_id", "player_id"], dropna=True)
    }

    for i, row in raw.iterrows():
        period = int(row["period_id"]) if pd.notna(row["period_id"]) else None
        pid = row["player_id"]
        kind = row["event_kind"]
        ts = float(row["timestamp"]) if pd.notna(row["timestamp"]) else np.nan
        if period is None or np.isnan(ts) or not isinstance(pid, str):
            continue

        cands = el_by_pp.get((period, pid))
        if cands is None or cands.empty:
            continue

        gaps = (cands["ts_s"] - ts).abs().values
        kind_targets = DFL_TO_SPADL.get(kind, set())

        # Tier 1: nearest kind-match within the looser kind-match window.
        if kind_targets:
            tm = cands["event_type"].isin(kind_targets).values
            elig = tm & (gaps <= KIND_MATCH_WINDOW_S)
            if elig.any():
                idx = np.where(elig)[0]
                j = int(idx[gaps[idx].argmin()])
                picked = cands.iloc[j]
                out.at[i, "sync_ts"] = float(picked["ts_s"])
                out.at[i, "sync_quality"] = "player_label"
                continue

        # Tier 2: nearest same-player event within the tighter window.
        elig = gaps <= ANY_MATCH_WINDOW_S
        if elig.any():
            idx = np.where(elig)[0]
            j = int(idx[gaps[idx].argmin()])
            picked = cands.iloc[j]
            out.at[i, "sync_ts"] = float(picked["ts_s"])
            out.at[i, "sync_quality"] = "player_only"
            continue

    # Median-drift fallback for rows with no per-event match.
    for period in sorted(out["period_id"].dropna().unique()):
        mask_period = out["period_id"] == period
        mask_matched = mask_period & out["sync_ts"].notna()
        mask_unmatched = mask_period & out["sync_ts"].isna()
        if not mask_unmatched.any():
            continue
        if mask_matched.any():
            median_drift = float(
                (out.loc[mask_matched, "sync_ts"]
                 - out.loc[mask_matched, "timestamp"]).median())
            out.loc[mask_unmatched, "sync_ts"] = (
                out.loc[mask_unmatched, "timestamp"] + median_drift)
            out.loc[mask_unmatched, "sync_quality"] = "median"

    # Anything still un-synced -> no_match (keep timestamp unchanged).
    final_unmatched = out["sync_ts"].isna()
    out.loc[final_unmatched, "sync_ts"] = out.loc[final_unmatched, "timestamp"]
    out.loc[final_unmatched, "sync_quality"] = "no_match"

    out["sync_drift"] = out["sync_ts"] - out["timestamp"]
    return out


def main():
    """
        python build_raw_gt_synced.py --raw_data_path ./data/dfl/raw --data_path ./data/dfl/processed/elastic --save_path ./data/dfl/processed/raw_gt_synced
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="./data/dfl/raw")
    parser.add_argument("--data_path", type=str, default="./data/dfl/processed/elastic")
    parser.add_argument("--save_path", type=str, default="./data/dfl/processed/raw_gt_synced")
    args = parser.parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    match_ids = sorted(d.name for d in Path(args.data_path).iterdir() if d.is_dir())

    summary_rows = []
    for match_id in match_ids:
        if not (Path(args.data_path) / match_id / "event.parquet").exists():
            print(f"  {match_id}: skipped (no elastic event.parquet)")
            continue
        raw = extract_match(match_id, args.raw_data_path)
        out = build_synced(raw, match_id, args.data_path)

        out.to_parquet(Path(args.save_path) / f"{match_id}.parquet")
        qdist = out["sync_quality"].value_counts().to_dict()
        print(f"  {match_id}: rows={len(out)}  "
              f"drift mean={out.sync_drift.mean():+.2f}s  q={qdist}")
        kc = out["event_kind"].value_counts().to_dict()
        summary_rows.append({"match": match_id, "total": len(out), **kc})

    if summary_rows:
        summary = pd.DataFrame(summary_rows).fillna(0)
        for col in summary.columns:
            if col != "match":
                summary[col] = summary[col].astype(int)
        print("\nPer-match event_kind distribution:")
        print(summary.to_string(index=False))
        summary.to_csv(Path(args.save_path) / "summary.csv", index=False)

if __name__ == "__main__":
    main()