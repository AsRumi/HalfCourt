"""
_fetcher_common.py

Shared helpers used by both data_fetcher.py (historical) and live_fetcher.py (live).
Nothing in this module makes assumptions about whether the game is finished.
"""

import json
import logging
import time
from pathlib import Path

from nba_api.stats.endpoints import BoxScoreTraditionalV3, BoxScoreSummaryV2

from halfcourt.exceptions import DataFetchError

log = logging.getLogger(__name__)

SLEEP_BETWEEN_CALLS = 0.6
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


# Cache Helpers

def save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    log.info("  Cached → %s", path)


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text())


# Stat helpers

def zero_stats() -> dict:
    return {
        "MIN": 0.0,
        "FGM": 0, "FGA": 0,
        "FG3M": 0, "FG3A": 0,
        "FTM": 0, "FTA": 0,
        "OREB": 0, "DREB": 0, "REB": 0,
        "AST": 0, "STL": 0, "BLK": 0,
        "TO": 0, "PF": 0, "PTS": 0,
    }


def col(row: dict, *candidates):
    for c in candidates:
        if c in row:
            return row[c]
    return None


def clean_columns(df):
    df = df.copy()
    if "firstName" in df.columns and "familyName" in df.columns:
        df["PLAYER_NAME"] = (
            df["firstName"].fillna("") + " " + df["familyName"].fillna("")
        ).str.strip()

    rename_map = {
        "teamId": "TEAM_ID",
        "personId": "PLAYER_ID",
        "points": "PTS",
        "minutes": "MIN",
        "teamTricode": "TEAM_ABBREVIATION",
        "position": "START_POSITION",
        "fieldGoalsMade": "FGM",
        "fieldGoalsAttempted": "FGA",
        "threePointersMade": "FG3M",
        "threePointersAttempted": "FG3A",
        "freeThrowsMade": "FTM",
        "freeThrowsAttempted": "FTA",
        "reboundsOffensive": "OREB",
        "reboundsDefensive": "DREB",
        "reboundsTotal": "REB",
        "assists": "AST",
        "steals": "STL",
        "blocks": "BLK",
        "turnovers": "TO",
        "foulsPersonal": "PF",
    }
    return df.rename(columns=rename_map)


def accumulate(agg: dict, row: dict) -> None:
    for key in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS"]:
        agg[key] += row.get(key, 0) or 0

    raw_min = row.get("MIN", "0:00")
    if isinstance(raw_min, str) and ":" in raw_min:
        mm, ss = raw_min.split(":")
        agg["MIN"] += int(mm) + int(ss) / 60
    else:
        agg["MIN"] += float(raw_min or 0)


def compute_pcts(p: dict) -> dict:
    p["FG_PCT"]  = round(p["FGM"] / p["FGA"],  3) if p["FGA"]  else None
    p["FG3_PCT"] = round(p["FG3M"] / p["FG3A"], 3) if p["FG3A"] else None
    p["FT_PCT"]  = round(p["FTM"] / p["FTA"],  3) if p["FTA"]  else None
    return p


def team_totals(players: list[dict]) -> dict:
    cols = ("FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS")
    totals = {c: sum(p.get(c) or 0 for p in players) for c in cols}
    totals["FG_PCT"]  = round(totals["FGM"] / totals["FGA"],  3) if totals["FGA"]  else None
    totals["FG3_PCT"] = round(totals["FG3M"] / totals["FG3A"], 3) if totals["FG3A"] else None
    totals["FT_PCT"]  = round(totals["FTM"] / totals["FTA"],  3) if totals["FTA"]  else None
    totals["bench_pts"] = sum(
        p.get("PTS") or 0 for p in players if not p.get("start_position")
    )
    return totals


# NBA API calls

def call_boxscore(game_id: str, period: int) -> tuple[list, list]:
    """Fetch one quarter's player + team rows. Retries with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bs = BoxScoreTraditionalV3(
                game_id=game_id,
                start_period=period,
                end_period=period,
                range_type=1,
                start_range=0,
                end_range=0,
            )
            p_df = clean_columns(bs.player_stats.get_data_frame())
            t_df = clean_columns(bs.team_stats.get_data_frame())
            return p_df.to_dict(orient="records"), t_df.to_dict(orient="records")
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning(
                "  Attempt %d/%d failed — game %s Q%d: %s — retrying in %.1fs",
                attempt, MAX_RETRIES, game_id, period, exc, wait,
            )
            time.sleep(wait)
    raise DataFetchError(
        f"Failed to fetch Q{period} for game {game_id} after {MAX_RETRIES} attempts"
    )


def fetch_context_features(game_id: str) -> dict:
    """
    Fetch game date, home/away identities, rest days, and back-to-back flags
    via BoxScoreSummaryV2.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            summary = BoxScoreSummaryV2(game_id=game_id)
            frames = summary.get_data_frames()
            time.sleep(SLEEP_BETWEEN_CALLS)

            game_summary = frames[0]
            if game_summary.empty:
                raise DataFetchError(f"No game summary for {game_id}")

            row = game_summary.iloc[0]
            game_date = str(row.get("GAME_DATE_EST", ""))[:10]

            line_score = frames[5]
            if len(line_score) < 2:
                raise DataFetchError(f"LineScore has <2 rows for {game_id}")

            home_id = int(row["HOME_TEAM_ID"])
            away_id = int(row["VISITOR_TEAM_ID"])

            home_row = line_score[line_score["TEAM_ID"] == home_id]
            away_row = line_score[line_score["TEAM_ID"] == away_id]

            if home_row.empty or away_row.empty:
                raise DataFetchError(
                    f"Could not match home/away in LineScore for {game_id}"
                )

            home_abbr = str(home_row.iloc[0].get("TEAM_ABBREVIATION", ""))
            away_abbr = str(away_row.iloc[0].get("TEAM_ABBREVIATION", ""))
            is_playoffs = 1 if str(game_id)[2] == "4" else 0

            return {
                "home_team_id":   home_id,
                "home_team_abbr": home_abbr,
                "away_team_id":   away_id,
                "away_team_abbr": away_abbr,
                "game_date":      game_date,
                "is_playoffs":    is_playoffs,
                "home_rest_days": 3,
                "away_rest_days": 3,
                "home_back2back": 0,
                "away_back2back": 0,
            }

        except DataFetchError:
            raise
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning(
                "  Attempt %d/%d failed — context features for %s: %s",
                attempt, MAX_RETRIES, game_id, exc,
            )
            time.sleep(wait)

    raise DataFetchError(
        f"Failed to fetch context features for game {game_id} after {MAX_RETRIES} attempts"
    )


# Halftime parsing

def get_team_period_pts(team_rows: list, team_id: int) -> int:
    return sum(row["PTS"] for row in team_rows if row["TEAM_ID"] == team_id)


def parse_halftime(raw: dict, game_id: str) -> dict:
    """
    Build the canonical halftime dict from raw Q1/Q2 player + team rows
    and a context block (final dict).  Works for both historical and live games —
    the caller is responsible for what lives in raw["final"].
    """
    final = raw["final"]
    home_id = final["home_team_id"]
    away_id = final["away_team_id"]

    q1_home = get_team_period_pts(raw["q1_team_stats"], home_id)
    q1_away = get_team_period_pts(raw["q1_team_stats"], away_id)
    q2_home = get_team_period_pts(raw["q2_team_stats"], home_id)
    q2_away = get_team_period_pts(raw["q2_team_stats"], away_id)

    halftime_score = {
        "home_pts":       q1_home + q2_home,
        "away_pts":       q1_away + q2_away,
        "score_diff":     (q1_home + q2_home) - (q1_away + q2_away),
        "q1_home_pts":    q1_home,
        "q1_away_pts":    q1_away,
        "q1_diff":        q1_home - q1_away,
        "q2_home_pts":    q2_home,
        "q2_away_pts":    q2_away,
        "q2_diff":        q2_home - q2_away,
        "momentum_shift": (q2_home - q2_away) - (q1_home - q1_away),
    }

    player_agg: dict[int, dict] = {}
    for row in raw["player_stats"]:
        pid = row["PLAYER_ID"]
        if pid not in player_agg:
            player_agg[pid] = {
                "player_id":      pid,
                "player_name":    row.get("PLAYER_NAME", f"Player {pid}"),
                "team_id":        row["TEAM_ID"],
                "team_abbr":      row.get("TEAM_ABBREVIATION"),
                "start_position": row.get("START_POSITION", ""),
                **zero_stats(),
            }
        accumulate(player_agg[pid], row)

    home_players, away_players = [], []
    for p in player_agg.values():
        p = compute_pcts(p)
        if p["team_id"] == home_id:
            home_players.append(p)
        else:
            away_players.append(p)

    return {
        "game_id":        game_id,
        "final":          final,
        "halftime_score": halftime_score,
        "home_team": {
            "team_id": home_id,
            "abbr":    final["home_team_abbr"],
            "players": home_players,
            "totals":  team_totals(home_players),
        },
        "away_team": {
            "team_id": away_id,
            "abbr":    final["away_team_abbr"],
            "players": away_players,
            "totals":  team_totals(away_players),
        },
    }
