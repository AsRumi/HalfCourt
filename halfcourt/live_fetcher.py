"""
live_fetcher.py

Fetches halftime data for a game currently in progress (or paused at half)
using the NBA CDN live boxscore endpoint:
  https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json

This endpoint returns cumulative stats at the current point in the game,
so we don't need to fetch Q1 and Q2 separately — at halftime the totals
are exactly what we need.

Caches snapshots to data/raw/live/ so repeated calls during the same break
are fast. Delete the file to force a fresh fetch.
"""

import logging
import time
from pathlib import Path

import requests

from halfcourt._fetcher_common import (
    SLEEP_BETWEEN_CALLS,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    save_cache,
    load_cache,
    zero_stats,
    compute_pcts,
    team_totals,
)
from halfcourt.exceptions import DataFetchError

log = logging.getLogger(__name__)

LIVE_RAW_DIR = Path("data/raw/live")
_CDN_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _cache_path(game_id: str) -> Path:
    return LIVE_RAW_DIR / f"{game_id}.json"


def _fetch_cdn(game_id: str) -> dict:
    url = _CDN_URL.format(game_id=game_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=10)
            r.raise_for_status()
            return r.json()["game"]
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning(
                "  Attempt %d/%d failed — CDN fetch for %s: %s — retrying in %.1fs",
                attempt, MAX_RETRIES, game_id, exc, wait,
            )
            time.sleep(wait)
    raise DataFetchError(
        f"Failed to fetch live CDN boxscore for game {game_id} after {MAX_RETRIES} attempts"
    )


def _parse_player(raw_player: dict, team_id: int, team_abbr: str) -> dict:
    """Convert a CDN player dict into the canonical per-player format."""
    s = raw_player.get("statistics", {})
    is_starter = raw_player.get("starter", "0") == "1"

    p = {
        "player_id":      raw_player["personId"],
        "player_name":    raw_player.get("name", ""),
        "team_id":        team_id,
        "team_abbr":      team_abbr,
        "start_position": raw_player.get("position", "") if is_starter else "",
        # Stat fields matching zero_stats() keys
        "MIN":  0.0,
        "FGM":  s.get("fieldGoalsMade", 0),
        "FGA":  s.get("fieldGoalsAttempted", 0),
        "FG3M": s.get("threePointersMade", 0),
        "FG3A": s.get("threePointersAttempted", 0),
        "FTM":  s.get("freeThrowsMade", 0),
        "FTA":  s.get("freeThrowsAttempted", 0),
        "OREB": s.get("reboundsOffensive", 0),
        "DREB": s.get("reboundsDefensive", 0),
        "REB":  s.get("reboundsTotal", 0),
        "AST":  s.get("assists", 0),
        "STL":  s.get("steals", 0),
        "BLK":  s.get("blocks", 0),
        "TO":   s.get("turnovers", 0),
        "PF":   s.get("foulsPersonal", 0),
        "PTS":  s.get("points", 0),
    }

    # Parse ISO duration "PT16M55.30S" → decimal minutes
    raw_min = s.get("minutes", "PT0M0.00S")
    try:
        inner = raw_min.lstrip("PT")
        if "M" in inner:
            m_part, s_part = inner.split("M")
            p["MIN"] = float(m_part) + float(s_part.rstrip("S")) / 60
        else:
            p["MIN"] = float(inner.rstrip("S")) / 60
    except (ValueError, AttributeError):
        p["MIN"] = 0.0

    return compute_pcts(p)


def _parse_team(raw_team: dict) -> tuple[int, str, list[dict]]:
    """Return (team_id, team_abbr, parsed_players)."""
    team_id = raw_team["teamId"]
    team_abbr = raw_team["teamTricode"]
    players = [
        _parse_player(p, team_id, team_abbr)
        for p in raw_team.get("players", [])
        if p.get("played") == "1"
    ]
    return team_id, team_abbr, players


def _period_score(raw_team: dict, period: int) -> int:
    for p in raw_team.get("periods", []):
        if p["period"] == period:
            return p.get("score", 0)
    return 0


def _build_halftime_dict(game: dict, game_id: str) -> dict:
    home_raw = game["homeTeam"]
    away_raw = game["awayTeam"]

    home_id, home_abbr, home_players = _parse_team(home_raw)
    away_id, away_abbr, away_players = _parse_team(away_raw)

    q1_home = _period_score(home_raw, 1)
    q1_away = _period_score(away_raw, 1)
    q2_home = _period_score(home_raw, 2)
    q2_away = _period_score(away_raw, 2)

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

    is_playoffs = 1 if str(game_id)[2] == "4" else 0

    # game_date from gameTimeUTC: "2026-04-21T23:00:00Z" → "2026-04-21"
    game_date = game.get("gameTimeUTC", "")[:10]

    final = {
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

    return {
        "game_id":        game_id,
        "is_live":        True,
        "final":          final,
        "halftime_score": halftime_score,
        "home_team": {
            "team_id": home_id,
            "abbr":    home_abbr,
            "players": home_players,
            "totals":  team_totals(home_players),
        },
        "away_team": {
            "team_id": away_id,
            "abbr":    away_abbr,
            "players": away_players,
            "totals":  team_totals(away_players),
        },
    }


def fetch_live_game(game_id: str) -> dict:
    """
    Fetch cumulative halftime stats for a live (or mid-game paused) game
    from the NBA CDN live boxscore endpoint.

    The returned dict has the same shape as fetch_game() from data_fetcher.py,
    except the "final" block contains only context fields — home_win and
    final_margin are absent because the game is not over.

    Caches the snapshot to data/raw/live/{game_id}.json. Delete the file
    to force a fresh fetch.

    Args:
        game_id: NBA game ID string, e.g. "0042500112"

    Returns:
        Halftime dict ready for engineer_features(..., live=True).

    Raises:
        DataFetchError on network failure after max retries.
    """
    path = _cache_path(game_id)
    if path.exists():
        log.info("Live cache hit - game %s", game_id)
        return load_cache(path)

    log.info("Fetching live halftime data for game %s from CDN…", game_id)
    game = _fetch_cdn(game_id)
    time.sleep(SLEEP_BETWEEN_CALLS)

    halftime = _build_halftime_dict(game, game_id)

    save_cache(path, halftime)
    return halftime
