#!/bin/bash
set -e  # 에러 나면 멈춤

python elastic/convert_elastic.py --data_dir ./data/dfl/raw --save_dir ./data/dfl/processed/elastic --n_jobs -1


python build_raw_gt_synced.py \
  --raw_data_path ./data/dfl/raw \
  --data_path ./data/dfl/processed/elastic \
  --save_path ./data/dfl/processed/raw_gt_synced

python train.py \
  --data_path ./data/dfl/processed \
  --cache_path ./data/dfl/ml \
  --save_path ./data/dfl/ml/predictions