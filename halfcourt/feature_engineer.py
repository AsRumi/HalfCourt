import json
import os
import pandas as pd
from halfcourt.exceptions import FeatureEngineeringError


REQUIRED_FINAL_FIELDS_CONTEXT = [
    "home_team_abbr", "away_team_abbr",
    "game_date", "is_playoffs", "home_rest_days", "away_rest_days",
    "home_back2back", "away_back2back",
]

REQUIRED_FINAL_FIELDS_LABELS = [
    "home_win", "final_margin",
]

# Kept for backwards compatibility with any external callers
REQUIRED_FINAL_FIELDS = REQUIRED_FINAL_FIELDS_CONTEXT + REQUIRED_FINAL_FIELDS_LABELS

REQUIRED_HALFTIME_FIELDS = [
    "score_diff", "q1_diff", "q2_diff", "momentum_shift",
]

REQUIRED_TOTALS_FIELDS = [
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "TO", "PF", "PTS", "bench_pts",
]


def _validate_fields(source: dict, required: list, context: str) -> None:
    missing = [f for f in required if f not in source]
    if missing:
        raise FeatureEngineeringError(
            f"Missing required fields in {context}: {missing}"
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _compute_efg(fgm: int, fg3m: int, fga: int) -> float:
    return _safe_divide(fgm + 0.5 * fg3m, fga)


def _compute_ts(pts: int, fga: int, fta: int) -> float:
    return _safe_divide(pts, 2 * (fga + 0.44 * fta))


def _compute_pace(fga: int, oreb: int, to: int, fta: int, minutes: float = 24.0) -> float:
    # Possessions estimate: FGA - OREB + TO + (FTA * 0.44)
    possessions = fga - oreb + to + (fta * 0.44)
    return round(possessions * (48.0 / minutes), 4)


def _find_star_player(players: list) -> dict:
    if not players:
        raise FeatureEngineeringError("Player list is empty - cannot determine star player.")
    return max(players, key=lambda p: (p.get("PTS", 0), p.get("AST", 0)))


def _count_starters_with_fouls(players: list, foul_threshold: int = 3) -> int:
    return sum(
        1 for p in players
        if p.get("start_position", "") != "" and p.get("PF", 0) >= foul_threshold
    )


def _compute_bench_ratio(bench_pts: int, total_pts: int) -> float:
    return _safe_divide(bench_pts, total_pts)


def engineer_features(raw_game: dict, live: bool = False) -> dict:
    """
    Transform a single raw game JSON dict into a flat feature dict.

    Args:
        raw_game: dict loaded from data/raw/{game_id}.json or produced by live_fetcher
        live:     when True, skips label validation and omits home_win / final_margin
                  from the output (the game is not over yet)

    Returns:
        Flat dict with identifiers, all feature columns, and - unless live=True -
        the label columns home_win and final_margin.

    Raises:
        FeatureEngineeringError: if any required field is missing or malformed.
    """
    try:
        game_id = raw_game.get("game_id")
        if not game_id:
            raise FeatureEngineeringError("Missing 'game_id' in raw game dict.")

        final = raw_game.get("final", {})
        halftime = raw_game.get("halftime_score", {})
        home = raw_game.get("home_team", {})
        away = raw_game.get("away_team", {})

        _validate_fields(final, REQUIRED_FINAL_FIELDS_CONTEXT, "final")
        if not live:
            _validate_fields(final, REQUIRED_FINAL_FIELDS_LABELS, "final")
        _validate_fields(halftime, REQUIRED_HALFTIME_FIELDS, "halftime_score")

        home_totals = home.get("totals", {})
        away_totals = away.get("totals", {})
        _validate_fields(home_totals, REQUIRED_TOTALS_FIELDS, "home_team.totals")
        _validate_fields(away_totals, REQUIRED_TOTALS_FIELDS, "away_team.totals")

        home_players = home.get("players", [])
        away_players = away.get("players", [])
        if not home_players or not away_players:
            raise FeatureEngineeringError(f"Game {game_id} has empty player list for home or away team.")

        # --- Star players ---
        home_star = _find_star_player(home_players)
        away_star = _find_star_player(away_players)

        home_star_foul_flag = 1 if home_star.get("PF", 0) >= 3 else 0
        away_star_foul_flag = 1 if away_star.get("PF", 0) >= 3 else 0

        # --- Foul trouble ---
        home_starters_3foul = _count_starters_with_fouls(home_players)
        away_starters_3foul = _count_starters_with_fouls(away_players)

        # --- Shooting efficiency ---
        home_efg = _compute_efg(home_totals["FGM"], home_totals["FG3M"], home_totals["FGA"])
        away_efg = _compute_efg(away_totals["FGM"], away_totals["FG3M"], away_totals["FGA"])
        home_ts = _compute_ts(home_totals["PTS"], home_totals["FGA"], home_totals["FTA"])
        away_ts = _compute_ts(away_totals["PTS"], away_totals["FGA"], away_totals["FTA"])

        # --- Pace ---
        home_pace = _compute_pace(
            home_totals["FGA"], home_totals["OREB"], home_totals["TO"], home_totals["FTA"]
        )
        away_pace = _compute_pace(
            away_totals["FGA"], away_totals["OREB"], away_totals["TO"], away_totals["FTA"]
        )

        # --- Rebounding ---
        home_oreb_pct = _safe_divide(
            home_totals["OREB"], home_totals["OREB"] + away_totals["DREB"]
        )
        away_oreb_pct = _safe_divide(
            away_totals["OREB"], away_totals["OREB"] + home_totals["DREB"]
        )
        reb_diff = home_totals["REB"] - away_totals["REB"]

        # --- Ball control ---
        home_ast_to_ratio = _safe_divide(home_totals["AST"], home_totals["TO"])
        away_ast_to_ratio = _safe_divide(away_totals["AST"], away_totals["TO"])

        # --- Bench ---
        home_bench_ratio = _compute_bench_ratio(home_totals["bench_pts"], home_totals["PTS"])
        away_bench_ratio = _compute_bench_ratio(away_totals["bench_pts"], away_totals["PTS"])

        base = {
            # Identifiers
            "game_id": game_id,
            "game_date": final["game_date"],
            "home_team_abbr": final["home_team_abbr"],
            "away_team_abbr": final["away_team_abbr"],

            # Score state
            "ht_score_diff": int(halftime["score_diff"]),
            "q1_diff": int(halftime["q1_diff"]),
            "q2_diff": int(halftime["q2_diff"]),
            "momentum_shift": int(halftime["momentum_shift"]),

            # Shooting efficiency
            "home_fg_pct": round(float(home_totals["FG_PCT"] or 0.0), 4),
            "home_3p_pct": round(float(home_totals["FG3_PCT"] or 0.0), 4),
            "home_ft_pct": round(float(home_totals["FT_PCT"] or 0.0), 4),
            "home_efg_pct": home_efg,
            "home_ts_pct": home_ts,
            "away_fg_pct": round(float(away_totals["FG_PCT"] or 0.0), 4),
            "away_3p_pct": round(float(away_totals["FG3_PCT"] or 0.0), 4),
            "away_ft_pct": round(float(away_totals["FT_PCT"] or 0.0), 4),
            "away_efg_pct": away_efg,
            "away_ts_pct": away_ts,

            # Ball control
            "home_to": int(home_totals["TO"]),
            "away_to": int(away_totals["TO"]),
            "home_ast_to_ratio": home_ast_to_ratio,
            "away_ast_to_ratio": away_ast_to_ratio,

            # Rebounding
            "home_oreb_pct": home_oreb_pct,
            "away_oreb_pct": away_oreb_pct,
            "reb_diff": reb_diff,

            # Pace
            "home_pace_est": home_pace,
            "away_pace_est": away_pace,

            # Bench
            "home_bench_pts": int(home_totals["bench_pts"]),
            "away_bench_pts": int(away_totals["bench_pts"]),
            "home_bench_ratio": home_bench_ratio,
            "away_bench_ratio": away_bench_ratio,

            # Foul trouble
            "home_starters_3foul": home_starters_3foul,
            "away_starters_3foul": away_starters_3foul,
            "foul_diff": int(home_totals["PF"]) - int(away_totals["PF"]),

            # Star player
            "home_star1_pts": int(home_star.get("PTS", 0)),
            "home_star1_ast": int(home_star.get("AST", 0)),
            "home_star_foul_flag": home_star_foul_flag,
            "away_star1_pts": int(away_star.get("PTS", 0)),
            "away_star1_ast": int(away_star.get("AST", 0)),
            "away_star_foul_flag": away_star_foul_flag,

            # Context
            "is_home": 1,
            "home_rest_days": int(final["home_rest_days"]),
            "away_rest_days": int(final["away_rest_days"]),
            "home_back2back": int(final["home_back2back"]),
            "away_back2back": int(final["away_back2back"]),
            "is_playoffs": int(final["is_playoffs"]),
        }

        if not live:
            base["home_win"] = int(final["home_win"])
            base["final_margin"] = int(final["final_margin"])

        return base

    except FeatureEngineeringError:
        raise
    except Exception as e:
        raise FeatureEngineeringError(
            f"Unexpected error engineering features for game {raw_game.get('game_id', 'unknown')}: {e}"
        ) from e


def build_training_dataset(raw_dir: str, output_path: str) -> pd.DataFrame:
    """
    Process all JSON files in raw_dir into a training dataset.
    Writes both a Parquet file (for the model pipeline) and a CSV file
    (for inspection) to the same directory.

    Args:
        raw_dir:     path to data/raw/ containing one JSON file per game
        output_path: path to write the output Parquet file (CSV written alongside it)

    Returns:
        The processed DataFrame.

    Raises:
        FeatureEngineeringError: if raw_dir does not exist or no JSON files are found.
    """
    if not os.path.isdir(raw_dir):
        raise FeatureEngineeringError(f"Raw data directory not found: {raw_dir}")

    json_files = sorted(
        f for f in os.listdir(raw_dir) if f.endswith(".json")
    )
    if not json_files:
        raise FeatureEngineeringError(f"No JSON files found in {raw_dir}")

    rows = []
    skipped = []

    for filename in json_files:
        filepath = os.path.join(raw_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_game = json.load(f)
            row = engineer_features(raw_game)
            rows.append(row)
        except FeatureEngineeringError as e:
            skipped.append((filename, str(e)))
        except json.JSONDecodeError as e:
            skipped.append((filename, f"JSON decode error: {e}"))

    if not rows:
        raise FeatureEngineeringError("No games were successfully processed.")

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write CSV for inspection
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    df.to_csv(csv_path, index=False)

    # Parquet for model pipeline
    try:
        df.to_parquet(output_path, index=False)
        print(f"Parquet written : {output_path}")
    except ImportError:
        print("Parquet skipped : pyarrow not installed (run pip install pyarrow)")

    print(f"CSV written     : {csv_path}")
    print(f"  Rows          : {len(df)}")
    print(f"  Columns       : {len(df.columns)}")

    if skipped:
        print(f"  Skipped       : {len(skipped)} file(s)")
        for fname, reason in skipped:
            print(f"    - {fname}: {reason}")

    return df


if __name__ == "__main__":
    import yaml

    # Configure paths in config.yaml: default paths are data/raw and data/processed
    config_path = "config.yaml"
    raw_dir = "data/raw"
    output_path = "data/processed/halfcourt_training.parquet"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        raw_dir = config.get("data", {}).get("raw_cache_dir", raw_dir)
        processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
        filename = config.get("data", {}).get("processed_filename", "halfcourt_training.parquet")
        output_path = os.path.join(processed_dir, filename)

    print(f"Reading raw game files from : {raw_dir}")
    print(f"Output path                 : {output_path}")
    print("-" * 48)

    df = build_training_dataset(raw_dir, output_path)
