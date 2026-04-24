import os
import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from halfcourt.exceptions import ModelNotFoundError


# Config loader

def _load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


FEATURE_COLS = [
    "ht_score_diff", "q1_diff", "q2_diff", "momentum_shift",
    "home_fg_pct", "home_3p_pct", "home_ft_pct", "home_efg_pct", "home_ts_pct",
    "away_fg_pct", "away_3p_pct", "away_ft_pct", "away_efg_pct", "away_ts_pct",
    "home_to", "away_to", "home_ast_to_ratio", "away_ast_to_ratio",
    "home_oreb_pct", "away_oreb_pct", "reb_diff",
    "home_pace_est", "away_pace_est",
    "home_bench_pts", "away_bench_pts", "home_bench_ratio", "away_bench_ratio",
    "home_starters_3foul", "away_starters_3foul", "foul_diff",
    "home_star1_pts", "home_star1_ast", "home_star_foul_flag",
    "away_star1_pts", "away_star1_ast", "away_star_foul_flag",
    "is_home", "home_rest_days", "away_rest_days",
    "home_back2back", "away_back2back", "is_playoffs",
]

LABEL_COL = "home_win"
MARGIN_COL = "final_margin"


# DNN architecture

class HalfCourtDNN(nn.Module):
    """
    Multi-head DNN:
      Head 1 (classification): sigmoid -> win_probability
      Head 2 (regression):     linear  -> predicted_margin
    """

    def __init__(self, input_dim: int, hidden_layers: list, dropout: float):
        super().__init__()
        self.shared = self._build_shared(input_dim, hidden_layers, dropout)
        final_dim = hidden_layers[-1]
        self.win_head = nn.Linear(final_dim, 1)
        self.margin_head = nn.Linear(final_dim, 1)

    def _build_shared(self, input_dim: int, hidden_layers: list, dropout: float) -> nn.Sequential:
        layers = []
        in_dim = input_dim
        for out_dim in hidden_layers:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        shared = self.shared(x)
        win_prob = torch.sigmoid(self.win_head(shared)).squeeze(1)
        margin = self.margin_head(shared).squeeze(1)
        return win_prob, margin


# Sklearn-compatible wrapper

class SklearnDNNWrapper:
    """
    Wrap HalfCourtDNN to expose .fit() and .predict_proba() so the rest
    of the pipeline treats all three models identically.
    """

    def __init__(self, cfg: dict, input_dim: int):
        mlp_cfg = cfg["model"]["mlp"]
        self.epochs = mlp_cfg["epochs"]
        self.lr = mlp_cfg["learning_rate"]
        self.batch_size = mlp_cfg["batch_size"]
        self.margin_lambda = cfg["model"]["mlp"].get("margin_lambda", 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.background = None
        self.model = HalfCourtDNN(
            input_dim=input_dim,
            hidden_layers=mlp_cfg["hidden_layers"],
            dropout=mlp_cfg["dropout"],
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, margins: np.ndarray = None):
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
        self.background = X_scaled[bg_idx]
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if margins is None:
            margins = np.zeros(len(y), dtype=np.float32)
        m_tensor = torch.tensor(margins.astype(np.float32))

        dataset = TensorDataset(
            torch.tensor(X_scaled),
            y_tensor,
            m_tensor,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = self._run_epoch(loader, optimizer, bce_loss, mse_loss)
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss:.4f}")

        self.model.eval()

    def _run_epoch(self, loader, optimizer, bce_loss, mse_loss) -> float:
        total_loss = 0.0
        for X_batch, y_batch, m_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            m_batch = m_batch.to(self.device)

            optimizer.zero_grad()
            win_prob, margin_pred = self.model(X_batch)
            loss = bce_loss(win_prob, y_batch) + self.margin_lambda * mse_loss(margin_pred, m_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            win_prob, _ = self.model(torch.tensor(X_scaled).to(self.device))
        probs = win_prob.cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            _, margin_pred = self.model(torch.tensor(X_scaled).to(self.device))
        return margin_pred.cpu().numpy()


# Logistic Regression wrapper

class ScaledLogisticRegression:
    """
    Bundles a StandardScaler with a LogisticRegression so predictor.py
    can call .predict_proba() on raw unscaled feature arrays with ease.
    """

    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))


# Model factory

def build_model(model_type: str, cfg: dict, input_dim: int):
    """
    Returns an untrained model object for the given model_type.
    All three expose .fit() and .predict_proba().
    """
    if model_type == "xgboost":
        xgb_cfg = cfg["model"]["xgboost"]
        return XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            eval_metric="logloss",
            random_state=cfg["model"]["random_state"],
        )
    elif model_type == "logistic_regression":
        lr_cfg = cfg["model"]["logistic_regression"]
        return LogisticRegression(
            max_iter=lr_cfg["max_iter"],
            C=lr_cfg["C"],
            random_state=cfg["model"]["random_state"],
        )
    elif model_type == "mlp":
        return SklearnDNNWrapper(cfg=cfg, input_dim=input_dim)
    else:
        raise ModelNotFoundError(f"Unknown model type: '{model_type}'. Choose xgboost, logistic_regression, or mlp.")


# Data loading + temporal split

def _load_dataset(cfg: dict):
    processed_path = os.path.join(
        cfg["data"]["processed_dir"],
        cfg["data"]["processed_filename"],
    )
    try:
        df = pd.read_parquet(processed_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processed dataset not found at '{processed_path}'. Run 'collect' first."
        )

    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def _temporal_split(df: pd.DataFrame, test_size: float):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def _extract_arrays(df: pd.DataFrame):
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)
    margins = df[MARGIN_COL].values.astype(np.float32) if MARGIN_COL in df.columns else None
    return X, y, margins


def _fit_model(model, model_type: str, X_train, y_train, margins_train, X_val, y_val):
    if model_type == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif model_type == "logistic_regression":
        model.fit(X_train, y_train)
    elif model_type == "mlp":
        model.fit(X_train, y_train, margins=margins_train)
    return model


def _save_model(model, model_type: str, cfg: dict):
    save_dir = cfg["model"]["saved_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_type}.joblib")
    try:
        joblib.dump(model, save_path)
    except Exception as e:
        raise IOError(f"Failed to save model to '{save_path}': {e}")
    print(f"Model saved -> {save_path}")


# SHAP extraction

def _compute_shap_importance(model, model_type: str, X_test: np.ndarray) -> dict:
    background = X_test[:100] if len(X_test) >= 100 else X_test

    if model_type == "xgboost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif model_type == "logistic_regression":
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_test)
    elif model_type == "mlp":
        def win_prob_fn(x):
            return model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(win_prob_fn, background)
        shap_values = explainer.shap_values(X_test, nsamples=100)
    else:
        return {}

    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(zip(FEATURE_COLS, mean_abs))


# Evaluation report

def _print_evaluation(model_type, model, X_train, X_test, y_train, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    brier = brier_score_loss(y_test, proba)

    print("\n=== HalfCourt Model Evaluation ===")
    print(f"Model type     : {model_type}")
    print(f"Train samples  : {len(X_train)}")
    print(f"Test samples   : {len(X_test)}")
    print()
    print(f"Accuracy       : {accuracy:.3f}")
    print(f"AUC-ROC        : {auc:.3f}")
    print(f"F1 Score       : {f1:.3f}")
    print(f"Brier Score    : {brier:.3f}")
    print()

    shap_importance = _compute_shap_importance(model, model_type, X_test)
    sorted_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    print("Top 10 Features by SHAP importance:")
    for rank, (feat, val) in enumerate(sorted_features, start=1):
        print(f"  {rank:2d}. {feat:<30} {val:.3f}")
    print()


# Public Entry API

def train_model(model_type: str):
    cfg = _load_config()
    df = _load_dataset(cfg)

    train_df, test_df = _temporal_split(df, cfg["model"]["test_size"])

    X_train, y_train, margins_train = _extract_arrays(train_df)
    X_test, y_test, _ = _extract_arrays(test_df)

    print(f"Training {model_type} - {len(X_train)} train / {len(X_test)} test samples...")

    model = build_model(model_type, cfg, input_dim=X_train.shape[1])
    model = _fit_model(model, model_type, X_train, y_train, margins_train, X_test, y_test)

    _print_evaluation(model_type, model, X_train, X_test, y_train, y_test)
    _save_model(model, model_type, cfg)
