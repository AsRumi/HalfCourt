"""
agent.py

Orchestrate the full HalfCourt pipeline:
  data_fetcher -> feature_engineer -> predictor -> prompt_builder
  -> commentary_engine -> output_formatter
"""

import logging

from halfcourt.data_fetcher import fetch_game
from halfcourt.live_fetcher import fetch_live_game
from halfcourt.feature_engineer import engineer_features
from halfcourt.predictor import predict
from halfcourt.prompt_builder import build_prompt
from halfcourt.commentary_engine import generate_commentary
from halfcourt.output_formatter import format_report

log = logging.getLogger(__name__)


class Agent:
    def run(self, game_id: str, model_type: str = "mlp") -> dict:
        """
        Run the full pipeline for a completed game.

        Args:
            game_id:    NBA game ID string, e.g. "0022300512"
            model_type: model to use for inference (default: "mlp")

        Returns:
            The report dict from output_formatter: {"markdown": str, "json": dict}
        """
        log.info("Fetching game data - %s", game_id)
        raw_game = fetch_game(game_id)

        log.info("Engineering features")
        feature_vector = engineer_features(raw_game)

        game_context = {
            "home_team":    raw_game["home_team"]["abbr"],
            "away_team":    raw_game["away_team"]["abbr"],
            "game_date":    raw_game["final"]["game_date"],
            "is_playoffs":  bool(raw_game["final"]["is_playoffs"]),
            "ht_home_pts":  raw_game["halftime_score"]["home_pts"],
            "ht_away_pts":  raw_game["halftime_score"]["away_pts"],
            "q1_home_pts":  raw_game["halftime_score"]["q1_home_pts"],
            "q1_away_pts":  raw_game["halftime_score"]["q1_away_pts"],
            "q2_home_pts":  raw_game["halftime_score"]["q2_home_pts"],
            "q2_away_pts":  raw_game["halftime_score"]["q2_away_pts"],
        }

        log.info("Running inference - model: %s", model_type)
        prediction = predict(feature_vector, model_type)

        log.info("Building prompt")
        prompt = build_prompt(feature_vector, prediction, game_context)

        log.info("Generating commentary")
        commentary = generate_commentary(prompt)

        log.info("Writing report")
        report = format_report(commentary, prediction, game_context)

        return report

    def run_live(self, game_id: str, model_type: str = "mlp") -> dict:
        """
        Run the full pipeline for a game currently paused at halftime.
        Skips final-score fetching; no outcome labels are expected or produced.

        Args:
            game_id:    NBA game ID string, e.g. "0042500112"
            model_type: model to use for inference (default: "mlp")

        Returns:
            The report dict from output_formatter: {"markdown": str, "json": dict}
        """
        log.info("Fetching live halftime data - %s", game_id)
        raw_game = fetch_live_game(game_id)

        log.info("Engineering features (live mode)")
        feature_vector = engineer_features(raw_game, live=True)

        game_context = {
            "home_team":    raw_game["home_team"]["abbr"],
            "away_team":    raw_game["away_team"]["abbr"],
            "game_date":    raw_game["final"]["game_date"],
            "is_playoffs":  bool(raw_game["final"]["is_playoffs"]),
            "is_live":      True,
            "ht_home_pts":  raw_game["halftime_score"]["home_pts"],
            "ht_away_pts":  raw_game["halftime_score"]["away_pts"],
            "q1_home_pts":  raw_game["halftime_score"]["q1_home_pts"],
            "q1_away_pts":  raw_game["halftime_score"]["q1_away_pts"],
            "q2_home_pts":  raw_game["halftime_score"]["q2_home_pts"],
            "q2_away_pts":  raw_game["halftime_score"]["q2_away_pts"],
        }

        log.info("Running inference - model: %s", model_type)
        prediction = predict(feature_vector, model_type)

        log.info("Building prompt")
        prompt = build_prompt(feature_vector, prediction, game_context)

        log.info("Generating commentary")
        commentary = generate_commentary(prompt)

        log.info("Writing report")
        report = format_report(commentary, prediction, game_context)

        return report
