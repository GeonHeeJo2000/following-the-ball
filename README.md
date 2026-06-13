# Event Detection

This repository implements an algorithm for **automatic event detection in football** using player/ball tracking data and event data, based on the three-stage cascade framework described in *"Following the Ball: Goal-Aware Fine-Grained Event Detection in Soccer Tracking Data"*.

## Headline Results

#### Experiment 1 (Public): 7 Matches (5 Train / 1 Valid / 1 Test)

* Open Play

| Model | Pass F1 | Cross F1 | Shot F1 | **Micro F1** |
|---|---:|---:|---:|---:|
| Rule-based | ? | ? | ? | ? |
| **XGBoost**  | **0.934** | **0.789** | **0.702** | **0.926** |

* Set-piece (Rule-based only)

| Label | GT | Pred | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| corner_kick | 2 | 2 | 1.000 | 1.000 | 1.000 |
| free_kick | 26 | 33 | 0.758 | 0.962 | 0.848 |
| goal_kick | 19 | 20 | 0.950 | 1.000 | 0.974 |
| kickoff | 6 | 6 | 1.000 | 1.000 | 1.000 |
| throw_in | 54 | 51 | 1.000 | 0.944 | 0.971 |
---

## Python Environment Setup

Python 3.10.19.

```bash
git clone https://github.com/GeonHeeJo2000/Event-Detection.git
cd Event-Detection
pip install -r requirements.txt
```

## How to Access Data

#### DFL (Public)

- Official dataset download:
  [DFL](https://springernature.figshare.com/articles/dataset/-An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)
- **Data directory layout:** place the 3 XML files for each match (matchinformation, events, positions) as follows.
  ```text
  data/
  └── dfl/
      └── raw/
          └── <MATCH_ID 1>
              ├── matchinformation.xml
              ├── events.xml
              └── positions.xml
          └── <MATCH_ID 2>
              ├── matchinformation.xml
              ├── events.xml
              └── positions.xml
  ```

## End-to-end pipeline

```bash
# 1. Convert to Elastic format
python elastic/convert_elastic.py --data_dir ./data/dfl/raw --save_dir ./data/dfl/processed/elastic --n_jobs -1

# 2. Extract ground truth
python build_raw_gt_synced.py --raw_data_path ./data/dfl/raw --data_path ./data/dfl/processed/elastic --save_path ./data/dfl/processed/raw_gt_synced

# 3. Build the ML dataset and train the Stage-3 classifier
python train.py --data_path ./data/dfl/processed --cache_path ./data/dfl/ml --save_path ./data/dfl/ml/predictions --model xgb
# Other models (Table 3 ablation): --model {tabpfn,catboost,tabnet,fttransformer,tabtransformer,all}

# 4. Evaluate and compare models
(Generate rule-based predictions)
python eval/run_predictions.py

(Stage-1 set-piece + Stage-3 kick: Table 1/2 "Ours")
python evaluate.py \
    --raw_data_path ./data/dfl/processed/raw_gt_synced \
    --kick_pred_path ./data/dfl/ml/predictions/xgb_predictions.parquet \
    --set_piece_pred_path ./data/dfl/ml/detection

(Stage-3 classifier ablation: Table 3)
python evaluate.py \
    --raw_data_path ./data/dfl/processed/raw_gt_synced \
    --xgb_pred ./data/dfl/ml/predictions/xgb_predictions.parquet \
    --tabpfn_pred ./data/dfl/ml/predictions/tabpfn_predictions.parquet \
    --catboost_pred ./data/dfl/ml/predictions/catboost_predictions.parquet \
    --tabnet_pred ./data/dfl/ml/predictions/tabnet_predictions.parquet \
    --ft_transformer_pred ./data/dfl/ml/predictions/fttransformer_predictions.parquet \
    --tab_transformer_pred ./data/dfl/ml/predictions/tabtransformer_predictions.parquet \
    --rule_pred ./data/dfl/ml/predictions/rule_predictions.parquet
```

---

## Overview

#### Pipeline
* Input: player/ball tracking data (**7 public DFL matches**) is used to automatically detect events in a football match.
* Pipeline: designed as a **three-stage cascade**, combining deterministic rule-based processing with ML-based contextual classification.
* Evaluation: Stage 1 and Stage 3 outputs are combined and jointly evaluated against ground-truth events via 1:1 greedy matching.

```text
┌───────────────────────────────────────────────────────────────────┐
│                            Tracking Data                          │
│                       (25 fps × 2 × 45 min)                       │
└────────────────────────────────┬──────────────────────────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Set-piece detection (Rule-based)                       │
│  ─────────────────────────────────────────────────────────────────│
│  • Ball State (ball position, height, sideline dist).             │
│                                                                   │
│  • Detects: KickOff, ThrowIn, CornerKick, FreeKick, GoalKick,     │
│             PenaltyKick.                                          │
│  • Key rules:                                                     │
│    – ball_z > 1.5 m + sideline_dist < 2 m        → ThrowIn        │
│    – ball at centre mark + half-line check       → KickOff        │
│    – ball at corner mark + dead ball state       → CornerKick     │
│    – ...                                         → ...            │
│                                                                   │
│  • Output: Set Piece Frame                                        │
└────────────────────────────────┬──────────────────────────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Possession detection (Rule-based)                      │
│  ─────────────────────────────────────────────────────────────────│
│  • Continuous-control state on the ball.                          │
│                                                                   │
│  • Detects: Loss of ball frame at event ball-control transition.  │
│  • Key rules:                                                     │
│    – R_PZ = 1.5 m, ε_θ=cos(10°), ε_v=5 m/s.          │
│    – ex) player A controls ball, ball leaves zone or changes      │
│      direction sharply → A loses ball at frame t                  │
│                                                                   │
│  • Output: ~2,000 loss frames for each match (frame, player).     │
└────────────────────────────────┬──────────────────────────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Open-play event classification                         │
│  ─────────────────────────────────────────────────────────────────│
│  • Classify into loss frame of event label                        │
│  • Set-piece frames inherit their Stage-1 label.                  │
│                                                                   │
│  • Detects: Open Play (Pass, Cross, Shot, None).                  │
│  • Two Classification Model:                                      │
│    • Rule-based classifier                                        │
│    • XGBoost                                                      │
│                                                                   │
│  • Output: Final event labels (set-piece + open-play).            │
└───────────────────────────────────────────────────────────────────┘
```

---

#### Why stage-wise separation works well
* Stage 1: Deterministic set-pieces
  - A throw-in is defined as the ball going out of play over the sideline and being restarted with a hand throw, and a corner kick is similarly well-defined by the ball's position in the corner arc together with the dead-ball state. A rule-based algorithm alone can reliably achieve an F1-score around 0.91, while an ML model would only end up learning this already-clear rule along with unnecessary noise.
* Stage 2: Deterministic touch frames
  - Detecting ball-loss frames is also largely deterministic. By setting a tracking radius around the ball and a direction-change threshold, touch events can be segmented cleanly. A few edge cases can be controlled via hyperparameter tuning.
* Stage 3: Ambiguous open-play situations
  - Classifying open-play situations is inherently ambiguous. Rules alone have limited power to distinguish a simple pass from a clearance, a long pass from a cross, or a low driven pass near the goal line from a shot. This requires precisely incorporating players' kinematic information and contextual signals, making it an area where machine learning (ML) models have a decisive advantage.

---

#### Models
* Both models below (Rule-based, XGBoost) rely on high-dimensional spatiotemporal information derived from **tracking data (ball and player position coordinates)** as their core features.

1. **Rule-based Detector**
      - A rule-based model that **references** the following key works:
        - **Vidal-Codina et al. (2022)**: *[Automatic event detection in football using tracking data](https://arxiv.org/abs/2202.00804)*
        - **Hyunsung Kim et al. (2024)**: *[PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories](https://arxiv.org/abs/2602.12080)*
      - **Inference time (per frame):**
        - **CPU:** ?

2. **XGBoost Classifier**
    - A classification model using 53 hand-engineered features designed from domain knowledge.
    - **Inference time (per frame):**
      - **CPU:** 1.11 ms
      - **GPU:** 5.53 ms

## Experiments

#### 4.1 Data Split

- **Train**: 5 matches
- **Valid**: 1 match
- **Test**: 1 match

| match_id | Home                  | Away                   | Usage |
|:--------:|-----------------------|------------------------|-------|
| J03WMX   | 1. FC Köln            | FC Bayern München      | Train |
| J03WN1   | VfL Bochum 1848       | Bayer 04 Leverkusen    | Train |
| J03WOH   | Fortuna Düsseldorf    | SSV Jahn Regensburg    | Train |
| J03WOY.  | Fortuna Düsseldorf    | F.C. Hansa Rostock.    | Train |
| J03WPY   | Fortuna Düsseldorf    | 1. FC Nürnberg         | Train |
| J03WQQ   | Fortuna Düsseldorf    | FC St. Pauli           | Valid |
| J03WR9   | Fortuna Düsseldorf    | 1. FC Kaiserslautern   | Test  |

---

#### 4.2 Rule-based Feature Set

In development....

---

### 4.3 XGBoost Feature Set (n = 53)

* The XGBoost model takes 53 numerical features as input, covering the state at the moment of ball loss, the past trajectory, and the ball's future movement.
* **Future features:** include the trajectory 1-2 seconds after the kick, so the model is optimized for **offline (post-hoc) classification** rather than real-time detection.
* **Receiver context features:** use a heuristic that treats the nearest teammate within a 30° cone in the ball's direction of travel as the "expected receiver".
* **Key signals:** `delta_dist_to_goal_1s` and `traj_curvature` are the strongest indicators for distinguishing Cross from Shot (a cross curves toward the goal area, while a shot follows a fast, straight trajectory toward the goal).

| Feature Group | Count | Key Features | Notes |
|:---|:---:|:---|:---|
| **Ball state** | 7 | `ball x/y/z`, `vx/vy`, speed, acceleration | 3D coordinates and 2D velocity vector |
| **Past 0.5s kinematics** | 4 | `past_speed_mean/max`, `past_accel/z_max` | History of ball-control intensity |
| **Future 1-2s trajectory** | 5 | `future_speed_max`, `future_air_frames`, `delta_dist_to_goal_1s` | Outcome of the ball after the kick (offline only) |
| **Trajectory shape** | 2 | `traj_curvature`, `ball_dir_stability` | Curvature and directional stability of the trajectory |
| **Pitch geometry** | 4 | `dist_to_goal`, `cos_to_goal`, `sideline_dist`, `rel_x` | Ball position relative to the attacking direction |
| **Tactical zones (binary)** | 4 | `in_attacking_half`, `in_shot_zone`, `in_cross_zone`, `in_pa` | Whether the ball is inside key attacking zones |
| **Kicker state** | 3 | `lp_x`, `lp_y`, `lp_speed` | Position and speed of the player making the kick |
| **Opponent context** | 3 | `nearest_opp_dist`, `n_opps_in_5m`, `n_attackers_in_pa` | Intensity of defensive pressure |
| **Attacking cone** | 3 | `team_in_cone15/30`, `opp_in_cone30` | Number of teammates/opponents within a forward 15°/30° cone |
| **Distance along travel direction** | 2 | `nearest_team_along`, `nearest_opp_along` | Distance to the nearest receiver/defender along the ball's travel direction |
| **Receiver context** | 3 | `rcv_speed`, `rcv_align`, `rcv_pressure` | State of the expected receiver (teammate within the 30° cone) |
| **Possession outcome prediction** | 4 | `time_to_gain`, `same_team_gain`, `opp_team_gain` | Outcome of the possession change after the kick |
| **Temporal context** | 2 | `time_since_prev_loss`, `time_since_prev_same` | Time elapsed since the previous ball-loss frame |
| **Set-piece flags** | 7 | `is_set_piece`, `sp_corner`, `sp_freekick`, `sp_throw_in`, etc. | Which set-piece type is currently active |

## Repository structure

```
🗂️ Event-Detection/
├── 🗂️ data
│   └── 🗂️ dfl
        ├── 🗂️ raw
            ├── 🗂️ DFL-MAT-J03WMX # Match ID
            │   ├── 📄 DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml       # Match Metadata
            │   ├── 📄 DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml             # Event Data
            │   └── 📄 DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml # Tracking Data
            └── 🗂️ Other Match ID ... 
        ├── 🗂️ processed
            ├── 🗂️ raw_gt_synced                      # GT with synchronized timestamp column
            │   ├── 📄 DFL-MAT-J03WMX.parquet
            │   ├── 📄 <Other Match ID>.parquet ...
            │   └── 📄 summary.csv.                   # numbers of each event types
            └── 🗂️ elastic
                ├── 🗂️ DFL-MAT-J03WMX # Match ID
                │   ├── 📄 teams.parquet             
                │   ├── 📄 event.parquet
                │   └── 📄 tracking.parquet
                └── 🗂️ Other Match ID ... 
        └── 🗂️ ml                                # ML feature parquets + checkpoints
            ├── 🗂️ detection
            │   ├── 🗂️ DFL-MAT-J03WMX 
            │   │   ├── 📄 open_play.parquet
            │   │   ├── 📄 possession.parquet
            │   │   └── 📄 set_piece.parquet
            │   └── 🗂️ Other Match ID ... 
            ├── 🗂️ kick                      
            │   ├── 📄 DFL-MAT-J03WMX.parquet
            │   ├── 📄 <Other Match ID>.parquet ...
            │   └── 📄 info.txt.                   # data info
            ├── 🗂️ predictions                      
            │   ├── 📄 xgb_predictions.parquet         # per-frame xgb predictions
            └── └── 📄 info.txt.                       # data info
│
├── 🗂️ elastic                     # Synchronization of tracking and event data
│   ├── 🗂️ sync                    # Synchronization Logic
│   ├── 🗂️ tools                   # Unified Spadl format
│   ├── 📄 convert_elastic.py      # Main synchronization script
│   └── ...
├── 🗂️ eval                       # evaluation & matching
│   ├── 📄 run_compare.py         # main entry
│   ├── 📄 run_dedup_eval.py      # dedup + paper-style matching
│   ├── 📄 run_raw_eval.py        # GT mapping primitives (sync_ts based)
│   ├── 📄 run_ml_eval.py
│   └── 📄 run_predictions.py
├── 🗂️ models 
│   ├── 🗂️ AutoEvent              # possession state machine
│   └── 🗂️ RuleBased              # rule-based event detector
├── 📄 build_raw_gt_synced.py # XML → raw GT synced parquet
│── 📄 train.py                   # train Stage-3 classifier (--model xgb/tabpfn/catboost/tabnet/fttransformer/tabtransformer/all)
│── 📄 evaluate.py                # evaluate kick/set-piece/custom-rule + Table 3 ablation (--xgb_pred, --tabpfn_pred, ...)
└── ...
```

## Affiliations

Research conducted by
**[University of Seoul CIDA Lab](https://cida.uos.ac.kr)**.
