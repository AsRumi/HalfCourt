"""
predictor.py

Load a serialized model and run inference on a feature vector.
Extract top-3 SHAP values to explain the prediction.
"""

import os
import sys
import functools
import joblib
import numpy as np
import shap
import torch
import yaml

from halfcourt.exceptions import ModelNotFoundError
from halfcourt.model_trainer import FEATURE_COLS, SklearnDNNWrapper, HalfCourtDNN
import halfcourt.model_trainer as _trainer_module


def _load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def _load_model(model_type: str, cfg: dict):
    save_dir = cfg["model"]["saved_dir"]
    path = os.path.join(save_dir, f"{model_type}.joblib")
    if not os.path.exists(path):
        raise ModelNotFoundError(
            f"No saved model found at '{path}'. Run 'train --model-type {model_type}' first."
        )
    # Patch 1: remap Colab's __main__ classes to halfcourt.model_trainer
    main_mod = sys.modules["__main__"]
    patched = []
    for name in ("SklearnDNNWrapper", "HalfCourtDNN"):
        if not hasattr(main_mod, name):
            setattr(main_mod, name, getattr(_trainer_module, name))
            patched.append(name)
    # Patch 2: force CUDA tensors to load on CPU
    original_torch_load = torch.load
    torch.load = functools.partial(original_torch_load, map_location=torch.device("cpu"))
    try:
        model = joblib.load(path)
        # Ensure the wrapper also uses CPU going forward
        if hasattr(model, "device"):
            model.device = torch.device("cpu")
        if hasattr(model, "model"):
            model.model = model.model.cpu()
        return model
    finally:
        torch.load = original_torch_load
        for name in patched:
            delattr(main_mod, name)


def _confidence_tier(win_probability: float) -> str:
    if win_probability > 0.70 or win_probability < 0.30:
        return "high"
    if win_probability > 0.55 or win_probability < 0.45:
        return "medium"
    return "low"


def _extract_shap_top3(model, model_type: str, X: np.ndarray) -> list[dict]:
    """
    Compute SHAP values for a single sample and return the top 3 features
    by absolute magnitude, each tagged with a direction.
    """
    if model_type == "xgboost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)[0]
    elif model_type == "logistic_regression":
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)[0]
    elif model_type == "mlp":
        def win_prob_fn(x):
            return model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(win_prob_fn, X)
        shap_values = explainer.shap_values(X, nsamples=100)[0]
    else:
        return []

    indexed = sorted(
        enumerate(shap_values), key=lambda x: abs(x[1]), reverse=True
    )[:3]

    return [
        {
            "feature":   FEATURE_COLS[i],
            "value":     round(float(shap_values[i]), 4),
            "direction": "favors_home" if shap_values[i] > 0 else "favors_away",
        }
        for i, _ in indexed
    ]


def _predict_margin(model, model_type: str, X: np.ndarray) -> int:
    """
    Return a predicted point margin (signed, positive = home wins).
    XGBoost and logistic regression don't have a margin head, so we
    derive a rough estimate from win probability: scale linearly over [-20, 20].
    """
    if model_type == "mlp":
        return int(round(float(model.predict_margin(X)[0])))

    prob = float(model.predict_proba(X)[0, 1])
    # Linear mapping: prob=1.0 → +20, prob=0.0 → -20
    return int(round((prob - 0.5) * 40))


def predict(feature_vector: dict, model_type: str) -> dict:
    """
    Run inference on a flat feature dict.

    Args:
        feature_vector: dict matching the schema in feature_engineer.py
        model_type:     "xgboost" | "logistic_regression" | "mlp"

    Returns:
        {
            "predicted_winner":  "home" | "away",
            "win_probability":   float,   # P(home wins), 0-1
            "predicted_margin":  int,     # signed, positive = home wins by N
            "confidence_tier":   "high" | "medium" | "low",
            "top_features": [
                {"feature": str, "value": float, "direction": "favors_home"|"favors_away"}
            ]
        }
    """
    cfg = _load_config()
    model = _load_model(model_type, cfg)

    X = np.array(
        [feature_vector[col] for col in FEATURE_COLS], dtype=np.float32
    ).reshape(1, -1)

    win_prob = float(model.predict_proba(X)[0, 1])
    predicted_winner = "home" if win_prob >= 0.5 else "away"
    predicted_margin = _predict_margin(model, model_type, X)
    confidence_tier = _confidence_tier(win_prob)
    top_features = _extract_shap_top3(model, model_type, X)

    return {
        "predicted_winner": predicted_winner,
        "win_probability":  round(win_prob, 4),
        "predicted_margin": predicted_margin,
        "confidence_tier":  confidence_tier,
        "top_features":     top_features,
    }
