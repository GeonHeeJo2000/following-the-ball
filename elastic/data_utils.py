import os
import numpy as np
import pandas as pd

from typing import List
from lxml import etree

def get_dfl_match_result(match_info_path):
    """
    Extracts match result from the XML tree.
    Returns: dict with home_team, guest_team, home_goals, guest_goals
    """
    tree = etree.parse(match_info_path)
    root = tree.getroot()
    
    match_info_node = None
    for child in root:
        if child.tag == "MatchInformation":
            match_info_node = child
            break
    
    if match_info_node is None:
        return None
    
    general_attrib = {}
    for subchild in match_info_node:
        if subchild.tag == "General":
            general_attrib = subchild.attrib
            break
    
    # 메타 정보
    competition_name = general_attrib.get('CompetitionName', '')
    season_name = general_attrib.get('Season', '')
    match_date = general_attrib.get('PlannedKickoffTime', '')
    
    # 결과 파싱 (예: "2:1" 형식)
    result_str = general_attrib.get('Result', '')
    home_team = general_attrib.get('HomeTeamName', '')
    guest_team = general_attrib.get('GuestTeamName', '')
    
    # 결과 문자열에서 골 수 추출
    if ':' in result_str:
        parts = result_str.split(':')
        try:
            home_goals = int(parts[0].strip())
            guest_goals = int(parts[1].strip())
        except ValueError:
            home_goals = None
            guest_goals = None
    else:
        home_goals = None
        guest_goals = None
    
    return {
        "competition_name": competition_name,
        "season_name": season_name,
        "match_date": match_date,
        'home_team': home_team,
        'guest_team': guest_team,
        'home_goals': home_goals,
        'guest_goals': guest_goals,
        'result_str': result_str
    }
    
def calc_match_result(match_id_list, data_path, provider):
    match_results = []


    if provider == "dfl":
        match_id_list = os.listdir(data_path)
        for match_id in match_id_list:
            match_path = os.path.join(data_path, match_id)

            if not os.path.isdir(match_path):
                continue

            file_name_info = next(
                (
                    fn for fn in os.listdir(match_path)
                    if "matchinformation" in fn.lower() or "spielinformationen" in fn.lower()
                ),
                None
            )
            if file_name_info is None:
                continue

            match_info_path = os.path.join(match_path, file_name_info)
            result = get_dfl_match_result(match_info_path)

            if result and result.get("home_goals") is not None:
                result["match_id"] = match_id
                match_results.append(result)       
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    print(f"Collected match results for {len(match_results)} matches.")
    return pd.DataFrame(match_results)

def generate_phase_records(traces):
    home_players = sorted([c[:-2] for c in traces.columns if c.startswith("H") and c.endswith("_x")])
    away_players = sorted([c[:-2] for c in traces.columns if c.startswith("A") and c.endswith("_x")])

    players = home_players + away_players
    player_x_cols = [f"{p}_x" for p in players]

    traces = traces.copy()
    traces["phase"] = 0
    
    play_records = []
    phase_records = []

    for p in players:
        valid_player_idx = traces[traces[f"{p}_x"].notna()].index
        f0 = valid_player_idx[0]
        f1 = valid_player_idx[-1]

        if len(traces.loc[f1, player_x_cols].dropna()) > 22:
            traces.loc[f1, [f"{p}_x", f"{p}_y"]] = np.nan
            play_records.append([p, f0, f1 - 1])
        else:
            play_records.append([p, f0, f1])

    play_records = pd.DataFrame(play_records, columns=["object", "start_frame", "end_frame"]).set_index("object")

    change_frames = play_records["start_frame"].tolist()
    change_frames.extend(
        [
            traces[traces["period_id"] == 2].index[0],
            #play_records["end_frame"].max(),
        ]
    )
    change_frames = list(set(change_frames))
    change_frames.sort()

    # for i, f0 in enumerate(change_frames[:-1]):
    for i, f0 in enumerate(change_frames[:]):
        # f1 = change_frames[i + 1] - 1
        if i == len(change_frames) - 1:
            f1 = traces.index[-1]
        else:
            f1 = change_frames[i + 1] - 1
        
        traces.loc[f0:f1, "phase"] = i + 1

        period_id = traces.loc[f0, "period_id"]
        start_time = round(traces.at[f0, "time"], 1)
        
        if f1 == traces.index[-1]:
            end_time = round(traces.at[f1, "time"], 1)
        else:
            end_time = round(traces.at[f1 + 1, "time"] - 0.1, 1)

        inplay_flags = traces.loc[f0:f1, player_x_cols].notna().any()
        player_codes = [c[:-2] for c in inplay_flags[inplay_flags].index]

        phase_records.append([i + 1, period_id, start_time, end_time, player_codes])

    header = ["phase", "period_id", "start_time", "end_time", "player_codes"]
    phase_records = pd.DataFrame(phase_records, columns=header).set_index("phase")
    return traces, phase_records

def assign_episode(ball_state: pd.Series, state: List[str]):
    """
        alive, interpolated, dead 상태 중 state에 해당하는 상태를 episode로 설정
    """
    
    is_state = ball_state.isin(state)

    # alive 구간의 시작점 (dead -> alive)
    alive_start = is_state & (~is_state.shift(fill_value=False))

    episode_id = alive_start.cumsum() # alive 구간 누적 카운트
    episode_id = episode_id.where(is_state, 0)

    return episode_id
        

def split_into_episodes(traces, state=["alive"]):
    """
        ball_state = alive인 구간을 하나의 episode로 설정
    """
    
    traces = traces.copy()
        
    if "phase" not in traces.columns:
        print("Looking for 'phase' column... If it doesn't exist, please run generate_phase_records() first.")
        traces, _ = generate_phase_records(traces)
        
    traces["episode"] = (
        traces
        .groupby(["period_id", "phase"])["ball_state"]
        .apply(lambda ball_state: assign_episode(ball_state, state=state))
        .reset_index(level=["period_id", "phase"], drop=True)
    )
        
    # 길이가 10 미만인 episode는 제외
    episode_lengths = traces[traces["episode"] > 0].groupby(["period_id", "phase", "episode"]).size()
    short_episodes = episode_lengths[episode_lengths < 10].index.tolist()
    for period_id, phase, episode in short_episodes:
        mask = (
            (traces["period_id"] == period_id) &
            (traces["phase"] == phase) &
            (traces["episode"] == episode)
        )
        traces.loc[mask, "episode"] = 0

    return traces