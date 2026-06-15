# Event Detection

This repository implements an algorithm for **automatic event detection in football** using player/ball tracking data and event data, based on the three-stage cascade framework described in *"Following the Ball: Goal-Aware Fine-Grained Event Detection in Soccer Tracking Data"*.

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
# Other models: --model {tabpfn,catboost,tabnet,fttransformer,tabtransformer,all}

# 4. Evaluate and compare models
(Generate rule-based predictions)
python eval/run_predictions.py

(Stage-1 set-piece + Stage-3 kick)
python evaluate.py \
    --raw_data_path ./data/dfl/processed/raw_gt_synced \
    --kick_pred_path ./data/dfl/ml/predictions/xgb_predictions.parquet \
    --set_piece_pred_path ./data/dfl/ml/detection

(Stage-3 classifier ablation)
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
│── 📄 evaluate.py                # evaluate kick/set-piece/custom-rule + Stage-3 classifier ablation (--xgb_pred, --tabpfn_pred, ...)
└── ...
```

## Affiliations

Research conducted by
**[University of Seoul CIDA Lab](https://cida.uos.ac.kr)**.
