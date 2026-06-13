"""Stage-3 open-play classifier training (Table 1 "Ours" + Table 3 ablation).

  train: 5 matches    valid: 1 match    test: 1 match

Models: xgb (default), tabpfn, catboost, tabnet, fttransformer, tabtransformer, all.
tabpfn requires the TabPFN client to be authenticated via the TABPFN_API_TOKEN
environment variable (https://www.tabpfn.com/).

Usage:
    python train.py --data_path ./data/dfl/processed --cache_path ./data/dfl/ml --save_path ./data/dfl/ml/predictions
    python train.py --data_path ./data/dfl/processed --cache_path ./data/dfl/ml --save_path ./data/dfl/ml/predictions --model all
"""
from __future__ import annotations
from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from dataset import KickDataset
from config import (
    KICK_FEATURE_COLS,
    KICK_LABELS,
    KICK_LABEL_TO_IDX,
)


def prepare_data(df: pd.DataFrame, feature_cols: list[str], label_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = df["label"].map(label_to_idx).to_numpy(dtype=int)

    # Compute class weights to handle imbalance
    counts = df["label"].value_counts()
    labels = list(label_to_idx.keys())
    w_per_label = {label: len(df) / (len(labels) * counts.get(label, 1)) for label in labels}
    weight = df["label"].map(w_per_label).to_numpy(dtype=float)

    return X, y, weight


def _print_results(out_df: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray, labels: list[str], label_to_idx: dict[str, int], model_name: str) -> None:
    print()
    print("Per-match per-frame F1:")
    for match_id in sorted(out_df["match_id"].dropna().unique()):
        sub = out_df[out_df.match_id == match_id]
        if len(sub) == 0:
            continue
        yy = sub["label"].map(label_to_idx).to_numpy()
        pp = sub["pred_label"].map(label_to_idx).to_numpy()
        report = classification_report(yy, pp, target_names=labels, digits=3, zero_division=0, output_dict=True)
        f1s = {label: report[label]["f1-score"] for label in labels}
        print(f"  {match_id}: " + "  ".join(f"{label[:4]}={f1s[label]:.3f}" for label in labels))

    print()
    print("=" * 65)
    print(f"{model_name} — test ({len(out_df['match_id'].dropna().unique())} matches):")
    print("=" * 65)
    print(classification_report(y_test, y_pred, target_names=labels, digits=3, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
    print(pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels]).to_string())


def train_xgb(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[str],
    label_to_idx: dict[str, int],
    output_dir: Path,
    model_name: str,
    extra_model_params: dict | None = None,
) -> pd.DataFrame:
    import xgboost as xgb

    X_train, y_train, weight_train = prepare_data(train_df, feature_cols, label_to_idx)
    X_valid, y_valid, _ = prepare_data(valid_df, feature_cols, label_to_idx)
    X_test, y_test, _ = prepare_data(test_df, feature_cols, label_to_idx)

    params = dict(
        objective="multi:softprob",
        num_class=len(labels),
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",
        random_state=42,
        device="cuda:0",
        n_jobs=-1,
        verbosity=0,
    )
    if extra_model_params:
        params.update(extra_model_params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=weight_train, eval_set=[(X_valid, y_valid)], verbose=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(output_dir) / f"{model_name}.json"
    model.save_model(model_path)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    out_df = test_df.copy()
    out_df["pred_label"] = [labels[p] for p in y_pred]
    for i, label in enumerate(labels):
        out_df[f"prob_{label}"] = y_proba[:, i]

    _print_results(out_df, y_test, y_pred, labels, label_to_idx, model_name)
    return out_df


def train_tabpfn(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[str],
    label_to_idx: dict[str, int],
    output_dir: Path,
    model_name: str,
    extra_model_params: dict | None = None,
) -> pd.DataFrame:
    import joblib
    from tabpfn_client import TabPFNClassifier, set_access_token

    tabpfn_token = os.environ.get("TABPFN_API_TOKEN")
    if not tabpfn_token:
        raise RuntimeError("TABPFN_API_TOKEN environment variable is not set. Get a token from https://www.tabpfn.com/ and export it.")
    set_access_token(tabpfn_token)

    X_train, y_train, _ = prepare_data(train_df, feature_cols, label_to_idx)
    X_test, y_test, _ = prepare_data(test_df, feature_cols, label_to_idx)

    params = dict(random_state=42)
    if extra_model_params:
        params.update(extra_model_params)

    model = TabPFNClassifier(**params)
    model.fit(X_train, y_train)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(output_dir) / f"{model_name}.pkl")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    out_df = test_df.copy()
    out_df["pred_label"] = [labels[p] for p in y_pred]
    for i, label in enumerate(labels):
        out_df[f"prob_{label}"] = y_proba[:, i]

    _print_results(out_df, y_test, y_pred, labels, label_to_idx, model_name)
    return out_df


def train_catboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[str],
    label_to_idx: dict[str, int],
    output_dir: Path,
    model_name: str,
    extra_model_params: dict | None = None,
) -> pd.DataFrame:
    from catboost import CatBoostClassifier

    X_train, y_train, weight_train = prepare_data(train_df, feature_cols, label_to_idx)
    X_valid, y_valid, _ = prepare_data(valid_df, feature_cols, label_to_idx)
    X_test, y_test, _ = prepare_data(test_df, feature_cols, label_to_idx)

    params = dict(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=100,
    )
    if extra_model_params:
        params.update(extra_model_params)

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=weight_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=50,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(str(Path(output_dir) / f"{model_name}.cbm"))

    y_pred = model.predict(X_test).flatten()
    y_proba = model.predict_proba(X_test)

    out_df = test_df.copy()
    out_df["pred_label"] = [labels[p] for p in y_pred]
    for i, label in enumerate(labels):
        out_df[f"prob_{label}"] = y_proba[:, i]

    _print_results(out_df, y_test, y_pred, labels, label_to_idx, model_name)
    return out_df


def train_tabnet(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[str],
    label_to_idx: dict[str, int],
    output_dir: Path,
    model_name: str,
    extra_model_params: dict | None = None,
) -> pd.DataFrame:
    from pytorch_tabnet.tab_model import TabNetClassifier

    X_train, y_train, weight_train = prepare_data(train_df, feature_cols, label_to_idx)
    X_valid, y_valid, _ = prepare_data(valid_df, feature_cols, label_to_idx)
    X_test, y_test, _ = prepare_data(test_df, feature_cols, label_to_idx)

    params = dict(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        seed=42,
        device_name="cuda",
        verbose=10,
    )
    if extra_model_params:
        params.update(extra_model_params)

    model = TabNetClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["accuracy"],
        weights=weight_train,
        max_epochs=200,
        patience=20,
        batch_size=4096,
        virtual_batch_size=512,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(str(Path(output_dir) / model_name))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    out_df = test_df.copy()
    out_df["pred_label"] = [labels[p] for p in y_pred]
    for i, label in enumerate(labels):
        out_df[f"prob_{label}"] = y_proba[:, i]

    _print_results(out_df, y_test, y_pred, labels, label_to_idx, model_name)
    return out_df


def _prep_pytorch_tabular(df: pd.DataFrame, feature_cols: list[str], label_to_idx: dict[str, int], target: str) -> pd.DataFrame:
    d = df[feature_cols + ["label"]].copy()
    d[target] = d["label"].map(label_to_idx).astype(int)
    return d.drop(columns=["label"])


def train_fttransformer(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[str],
    label_to_idx: dict[str, int],
    output_dir: Path,
    model_name: str,
    extra_model_params: dict | None = None,
) -> pd.DataFrame:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import FTTransformerConfig
    from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig

    TARGET = "label_idx"
    train_pt = _prep_pytorch_tabular(train_df, feature_cols, label_to_idx, TARGET)
    valid_pt = _prep_pytorch_tabular(valid_df, feature_cols, label_to_idx, TARGET)
    test_pt = _prep_pytorch_tabular(test_df, feature_cols, label_to_idx, TARGET)

    data_config = DataConfig(
        target=[TARGET],
        continuous_cols=feature_cols,
        categorical_cols=[],
        normalize_continuous_features=True,
    )
    trainer_config = TrainerConfig(
        batch_size=4096,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        early_stopping="valid_loss",
        early_stopping_patience=10,
        load_best=True,
        progress_bar="none",
        seed=42,
    )
    optimizer_config = OptimizerConfig(
        optimizer="Adam",
        lr_scheduler=None,
    )
    ft_params = dict(
        task="classification",
        num_attn_blocks=4,
        num_heads=8,
        input_embed_dim=64,
        attn_dropout=0.1,
        ff_dropout=0.1,
        learning_rate=1e-3,
        seed=42,
    )
    if extra_model_params:
        ft_params.update(extra_model_params)
    model_config = FTTransformerConfig(**ft_params)

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    model.fit(train=train_pt, validation=valid_pt)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(str(Path(output_dir) / model_name))

    pred_df = model.predict(test_pt)
    prob_cols = [f"{TARGET}_{i}_probability" for i in range(len(labels))]
    y_proba = pred_df[prob_cols].to_numpy()
    y_pred = pred_df[f"{TARGET}_prediction"].to_numpy().astype(int)
    y_test = test_df["label"].map(label_to_idx).to_numpy(dtype=int)

    out_df = test_df.copy()
    out_df["pred_label"] = [labels[p] for p in y_pred]
    for i, label in enumerate(labels):
        out_df[f"prob_{label}"] = y_proba[:, i]

    _print_results(out_df, y_test, y_pred, labels, label_to_idx, model_name)
    return out_df


def train_tabtransformer(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[str],
    label_to_idx: dict[str, int],
    output_dir: Path,
    model_name: str,
    extra_model_params: dict | None = None,
) -> pd.DataFrame:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import TabTransformerConfig
    from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig

    TARGET = "label_idx"
    train_pt = _prep_pytorch_tabular(train_df, feature_cols, label_to_idx, TARGET)
    valid_pt = _prep_pytorch_tabular(valid_df, feature_cols, label_to_idx, TARGET)
    test_pt = _prep_pytorch_tabular(test_df, feature_cols, label_to_idx, TARGET)

    data_config = DataConfig(
        target=[TARGET],
        continuous_cols=feature_cols,
        categorical_cols=[],
        normalize_continuous_features=True,
    )
    trainer_config = TrainerConfig(
        batch_size=4096,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        early_stopping="valid_loss",
        early_stopping_patience=10,
        load_best=True,
        progress_bar="none",
        seed=42,
    )
    optimizer_config = OptimizerConfig(
        optimizer="Adam",
        lr_scheduler=None,
    )
    tt_params = dict(
        task="classification",
        num_attn_blocks=4,
        num_heads=8,
        input_embed_dim=64,
        attn_dropout=0.1,
        ff_dropout=0.1,
        batch_norm_continuous_input=True,
        learning_rate=1e-3,
        seed=42,
    )
    if extra_model_params:
        tt_params.update(extra_model_params)
    model_config = TabTransformerConfig(**tt_params)

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    model.fit(train=train_pt, validation=valid_pt)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(str(Path(output_dir) / model_name))

    pred_df = model.predict(test_pt)
    prob_cols = [f"{TARGET}_{i}_probability" for i in range(len(labels))]
    y_proba = pred_df[prob_cols].to_numpy()
    y_pred = pred_df[f"{TARGET}_prediction"].to_numpy().astype(int)
    y_test = test_df["label"].map(label_to_idx).to_numpy(dtype=int)

    out_df = test_df.copy()
    out_df["pred_label"] = [labels[p] for p in y_pred]
    for i, label in enumerate(labels):
        out_df[f"prob_{label}"] = y_proba[:, i]

    _print_results(out_df, y_test, y_pred, labels, label_to_idx, model_name)
    return out_df


# model key -> (train_fn, model artifact name, prediction parquet name, extra params)
MODEL_REGISTRY = {
    "xgb": (train_xgb, "xgb_kick_model", "xgb_predictions.parquet", None),
    "tabpfn": (train_tabpfn, "tabpfn_kick_model", "tabpfn_predictions.parquet", None),
    "catboost": (train_catboost, "catboost_kick_model", "catboost_predictions.parquet", {"task_type": "GPU", "devices": "0"}),
    "tabnet": (train_tabnet, "tabnet_kick_model", "tabnet_predictions.parquet", None),
    "fttransformer": (train_fttransformer, "fttransformer_kick_model", "fttransformer_predictions.parquet", None),
    "tabtransformer": (train_tabtransformer, "tabtransformer_kick_model", "tabtransformer_predictions.parquet", None),
}


def run_kick(data_path, cache_path, save_path, model: str = "xgb") -> None:
    dataset = KickDataset(data_path=data_path, cache_path=cache_path, save_path=save_path)
    train_dataset = dataset.prepare_datasets(dataset.train_match_ids, split_name="train")
    valid_dataset = dataset.prepare_datasets(dataset.valid_match_ids, split_name="valid")
    test_dataset = dataset.prepare_datasets(dataset.test_match_ids, split_name="test")

    print(f"Kick train: {len(train_dataset)}  valid: {len(valid_dataset)}  test: {len(test_dataset)}")

    model_keys = list(MODEL_REGISTRY.keys()) if model == "all" else [model]
    for key in model_keys:
        train_fn, model_name, pred_filename, extra_params = MODEL_REGISTRY[key]
        out_df = train_fn(
            train_df=train_dataset,
            valid_df=valid_dataset,
            test_df=test_dataset,
            feature_cols=KICK_FEATURE_COLS,
            labels=KICK_LABELS,
            label_to_idx=KICK_LABEL_TO_IDX,
            output_dir=save_path,
            model_name=model_name,
            extra_model_params=extra_params,
        )
        out_path = Path(save_path) / pred_filename
        out_df.to_parquet(out_path)
        print(f"Saved → {out_path}")


def main() -> None:
    """
        python train.py --data_path ./data/dfl/processed --cache_path ./data/dfl/ml --save_path ./data/dfl/ml/predictions --model xgb
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=False, default=None, help="Path to trainning data (parquet files)")
    parser.add_argument("--cache_path", required=False, default="./cache", help="Path to cache intermediate data")
    parser.add_argument("--save_path", required=False, default=None, help="Path to save predictions (parquet file)")
    parser.add_argument("--model", required=False, default="xgb", choices=list(MODEL_REGISTRY.keys()) + ["all"],
                         help="Which Stage-3 classifier to train (default: xgb)")
    args = parser.parse_args()

    run_kick(args.data_path, args.cache_path, args.save_path, model=args.model)

if __name__ == "__main__":
    main()
