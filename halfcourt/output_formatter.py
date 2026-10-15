"""
output_formatter.py

Assembles the final report from commentary and prediction data.
Writes JSON and/or Markdown files to reports/{game_date}/{AWAY}_at_{HOME}.{ext}
based on config.
"""

import json
import os
from datetime import datetime

import yaml


def _load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def _build_markdown(commentary: str, prediction: dict, game_context: dict) -> str:
    winner_name = (
        game_context["home_team"] if prediction["predicted_winner"] == "home"
        else game_context["away_team"]
    )
    prob_pct = round(prediction["win_probability"] * 100, 1)
    raw_margin = prediction["predicted_margin"]

    if raw_margin is None:
        margin_row = "| Predicted Margin | *Not available for Logistic Regression - use XGBoost or MLP for margin prediction* |\n"
    else:
        margin_row = f"| Predicted Margin | {winner_name} by {abs(raw_margin)} pts |\n"

    top_features_lines = "\n".join(
        f"- **{f['feature']}** (SHAP: {f['value']:+.4f}) - favors "
        f"{'home' if f['direction'] == 'favors_home' else 'away'}"
        for f in prediction["top_features"]
    )

    live_badge = "**\U0001f534 LIVE PREDICTION**  \n" if game_context.get("is_live") else ""

    header = (
        f"# HalfCourt Report - {game_context['home_team']} vs {game_context['away_team']}\n\n"
        f"{live_badge}"
        f"**Date:** {game_context['game_date']}  \n"
        f"**Playoffs:** {'Yes' if game_context.get('is_playoffs') else 'No'}\n\n"
        f"---\n\n"
        f"## Model Prediction\n\n"
        f"| | |\n"
        f"|---|---|\n"
        f"| Predicted Winner | **{winner_name}** |\n"
        f"| Win Probability (home) | {prob_pct}% |\n"
        f"{margin_row}"
        f"| Confidence | {prediction['confidence_tier'].upper()} |\n\n"
        f"**Top Driving Features:**\n\n{top_features_lines}\n\n"
        f"---\n\n"
        f"## Analyst Commentary\n\n"
    )

    return header + commentary


def _build_json(commentary: str, prediction: dict, game_context: dict) -> dict:
    return {
        "game_context": game_context,
        "prediction":   prediction,
        "commentary":   commentary,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def _report_relpath(game_context: dict, ext: str) -> str:
    """Return '{game_date}/{AWAY}_at_{HOME}.{ext}', relative to reports_dir."""
    date = game_context.get("game_date") or "unknown_date"
    home = game_context.get("home_team", "HOME").replace(" ", "_")
    away = game_context.get("away_team", "AWAY").replace(" ", "_")
    return os.path.join(date, f"{away}_at_{home}.{ext}")


def _write_file(path: str, content: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Failed to write report to '{path}': {e}") from e


def format_report(commentary: str, prediction: dict, game_context: dict) -> dict:
    """
    Build and write the halftime report.

    Args:
        commentary:   validated commentary string from commentary_engine
        prediction:   output dict from predictor.predict()
        game_context: {"home_team": str, "away_team": str, "game_date": str, "is_playoffs": bool}

    Returns:
        {"markdown": str, "json": dict}
    """
    cfg = _load_config()
    reports_dir = cfg["output"]["reports_dir"]
    fmt = cfg["output"]["format"]

    markdown = _build_markdown(commentary, prediction, game_context)
    json_data = _build_json(commentary, prediction, game_context)

    if fmt in ("markdown", "both"):
        path = os.path.join(reports_dir, _report_relpath(game_context, "md"))
        _write_file(path, markdown)
        print(f"Markdown report -> {path}")

    if fmt in ("json", "both"):
        path = os.path.join(reports_dir, _report_relpath(game_context, "json"))
        _write_file(path, json.dumps(json_data, indent=2))
        print(f"JSON report     -> {path}")

    return {"markdown": markdown, "json": json_data}
