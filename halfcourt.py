"""HalfCourt CLI entrypoint."""
from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()


def cmd_predict(args: argparse.Namespace) -> None:
    """Run the full prediction + commentary pipeline on a single game."""
    print(f"Running prediction for game {args.game_id} using {args.model_type}...")
    from halfcourt.agent import Agent
    agent = Agent()
    agent.run(game_id=args.game_id, model_type=args.model_type)


def cmd_predict_live(args: argparse.Namespace) -> None:
    """Run the live prediction pipeline on a game currently paused at halftime."""
    print(f"Running live prediction for game {args.game_id} using {args.model_type}...")
    from halfcourt.agent import Agent
    agent = Agent()
    agent.run_live(game_id=args.game_id, model_type=args.model_type)


def cmd_collect(args: argparse.Namespace) -> None:
    """Collect halftime training data for the specified seasons."""
    import yaml
    seasons = args.seasons
    if not seasons:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        seasons = cfg["data"]["seasons"]
    print(f"Collecting data for seasons: {seasons}")
    from halfcourt.data_fetcher import fetch_seasons
    from halfcourt.feature_engineer import build_training_dataset
    import yaml as _yaml
    with open("config.yaml") as f:
        cfg = _yaml.safe_load(f)
    fetch_seasons(seasons)
    build_training_dataset(
        cfg["data"]["raw_cache_dir"],
        f"{cfg['data']['processed_dir']}/{cfg['data']['processed_filename']}",
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train a model on the collected training data."""
    print(f"Training {args.model_type} model...")
    from halfcourt.model_trainer import train_model
    train_model(model_type=args.model_type)


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare all three models on the test set."""
    print("Comparing all models on test set...")
    from halfcourt.model_trainer import train_model
    for model_type in ("xgboost", "logistic_regression", "mlp"):
        train_model(model_type=model_type)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halfcourt",
        description="HalfCourt — AI-Powered NBA Halftime Prediction & Commentary Agent.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # predict
    p_predict = subparsers.add_parser("predict", help="Run full pipeline on a single game.")
    p_predict.add_argument("--game-id", required=True, help="NBA game ID, e.g. 0022300512")
    p_predict.add_argument(
        "--model-type",
        default="mlp",
        choices=["xgboost", "logistic_regression", "mlp"],
        help="Model to use for prediction.",
    )
    p_predict.set_defaults(func=cmd_predict)

    # predict-live
    p_live = subparsers.add_parser(
        "predict-live",
        help="Run live prediction on a game currently paused at halftime.",
    )
    p_live.add_argument("--game-id", required=True, help="NBA game ID, e.g. 0022300512")
    p_live.add_argument(
        "--model-type",
        default="mlp",
        choices=["xgboost", "logistic_regression", "mlp"],
        help="Model to use for prediction.",
    )
    p_live.set_defaults(func=cmd_predict_live)

    # collect
    p_collect = subparsers.add_parser("collect", help="Collect training data for specified seasons.")
    p_collect.add_argument(
        "seasons",
        nargs="*",
        metavar="SEASON",
        help="Seasons to collect, e.g. 2021-22 2022-23. Defaults to config.yaml seasons.",
    )
    p_collect.set_defaults(func=cmd_collect)

    # train
    p_train = subparsers.add_parser("train", help="Train a model on collected data.")
    p_train.add_argument(
        "--model-type",
        default="mlp",
        choices=["xgboost", "logistic_regression", "mlp"],
        help="Model type to train.",
    )
    p_train.set_defaults(func=cmd_train)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare all three models on the test set.")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
