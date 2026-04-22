import time
import logging
from datetime import datetime
from pathlib import Path

from nba_api.stats.endpoints import BoxScoreTraditionalV3, LeagueGameFinder
from nba_api.stats.library.parameters import SeasonTypeAllStar

from halfcourt.exceptions import DataFetchError
from halfcourt._fetcher_common import (
    SLEEP_BETWEEN_CALLS,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    save_cache,
    load_cache,
    call_boxscore,
    fetch_context_features,
    parse_halftime,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def _cache_path(game_id: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{game_id}.json"


def _fetch_final_score_from_boxscore(game_id: str) -> tuple[dict, dict]:
    """Fetch full-game team totals from BoxScoreTraditionalV3 (all periods)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bs = BoxScoreTraditionalV3(
                game_id=game_id,
                start_period=1,
                end_period=10,
                range_type=0,
                start_range=0,
                end_range=0,
            )
            t_rows = bs.team_stats.get_data_frame().to_dict(orient="records")
            time.sleep(SLEEP_BETWEEN_CALLS)

            if len(t_rows) < 2:
                raise DataFetchError(
                    f"Expected 2 team rows, got {len(t_rows)} for game {game_id}")

            return t_rows[0], t_rows[1]

        except DataFetchError:
            raise
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning("  Attempt %d/%d failed - full boxscore for %s: %s",
                        attempt, MAX_RETRIES, game_id, exc)
            time.sleep(wait)

    raise DataFetchError(
        f"Failed to fetch full boxscore for game {game_id} after {MAX_RETRIES} attempts")


def _fetch_final_score(game_id: str) -> dict:
    """Combine full-game boxscore with context features to build the final block."""
    ctx = fetch_context_features(game_id)

    row0, row1 = _fetch_final_score_from_boxscore(game_id)

    home_id = ctx["home_team_id"]
    row0_team_id = row0.get("teamId") or row0.get("TEAM_ID")
    home_bs = row0 if row0_team_id == home_id else row1
    away_bs = row1 if row0_team_id == home_id else row0

    home_pts = home_bs.get("points") or home_bs.get("PTS") or 0
    away_pts = away_bs.get("points") or away_bs.get("PTS") or 0

    return {
        **ctx,
        "home_win":       1 if home_pts > away_pts else 0,
        "final_margin":   int(home_pts - away_pts),
        "home_final_pts": int(home_pts),
        "away_final_pts": int(away_pts),
    }


def _compute_rest(team_rows, game_date: str, current_game_id: str) -> tuple[int, int]:
    try:
        prior = team_rows[team_rows["GAME_ID"].astype(str) != str(current_game_id)]
        prior = prior.sort_values("GAME_DATE", ascending=False)
        if prior.empty or not game_date:
            return 3, 0
        last_date = datetime.strptime(prior.iloc[0]["GAME_DATE"], "%Y-%m-%d")
        curr_date = datetime.strptime(game_date, "%Y-%m-%d")
        rest = (curr_date - last_date).days - 1
        return max(rest, 0), 1 if rest == 0 else 0
    except Exception:
        return 3, 0


def _get_season_game_ids(season: str, season_type: str) -> list[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable="00",
            ).get_data_frames()[0]
            time.sleep(SLEEP_BETWEEN_CALLS)
            return df["GAME_ID"].unique().tolist()
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.warning("  Attempt %d/%d failed - game IDs for %s: %s",
                        attempt, MAX_RETRIES, season, exc)
            time.sleep(wait)
    raise DataFetchError(
        f"Failed to fetch game IDs for season {season} after {MAX_RETRIES} attempts")


def fetch_game(game_id: str, raw_dir: Path = RAW_DIR) -> dict:
    path = _cache_path(game_id, raw_dir)
    if path.exists():
        log.info("Cache hit - game %s", game_id)
        return load_cache(path)

    log.info("Fetching game %s from NBA API…", game_id)

    q1_p, q1_t = call_boxscore(game_id, 1)
    q2_p, q2_t = call_boxscore(game_id, 2)
    final = _fetch_final_score(game_id)

    raw = {
        "game_id":       game_id,
        "player_stats":  q1_p + q2_p,
        "q1_team_stats": q1_t,
        "q2_team_stats": q2_t,
        "final":         final,
    }
    halftime = parse_halftime(raw, game_id)
    save_cache(path, halftime)
    time.sleep(SLEEP_BETWEEN_CALLS)
    return halftime


def fetch_season(
    season: str,
    season_type: str = SeasonTypeAllStar.regular,
    raw_dir: Path = RAW_DIR,
) -> list[dict]:
    game_ids = _get_season_game_ids(season, season_type)
    log.info("Season %s - %d games found", season, len(game_ids))

    results = []
    for i, gid in enumerate(game_ids, 1):
        log.info("  [%d/%d] game %s", i, len(game_ids), gid)
        try:
            results.append(fetch_game(gid, raw_dir))
        except DataFetchError as e:
            log.warning("  Skipping game %s: %s", gid, e)

    log.info("Season %s complete - %d/%d games collected",
             season, len(results), len(game_ids))
    return results


def fetch_seasons(seasons):
    for season in seasons:
        fetch_season(season)


if __name__ == "__main__":
    games = fetch_seasons(["2014-15", "2015-16"])

    data = games[0] if games else None
    if not data:
        print("No games collected.")
        exit()

    home = data["home_team"]
    away = data["away_team"]
    final = data["final"]
    ht = data["halftime_score"]

    print(f"\n{'─'*60}")
    print(f"Game ID  : {data['game_id']}  |  Date: {final['game_date']}")
    print(f"Matchup  : {home['abbr']} (home) vs {away['abbr']} (away)")
    print(f"Playoffs : {'Yes' if final['is_playoffs'] else 'No'}  |  "
          f"Home rest: {final['home_rest_days']}d  |  "
          f"Away rest: {final['away_rest_days']}d  |  "
          f"Home B2B: {final['home_back2back']}  Away B2B: {final['away_back2back']}")
    print(f"\nHalftime : {home['abbr']} {ht['home_pts']} - {away['abbr']} {ht['away_pts']} "
          f"(diff: {ht['score_diff']:+d})")
    print(f"  Q1: {home['abbr']} {ht['q1_home_pts']} - {away['abbr']} {ht['q1_away_pts']} "
          f"(diff: {ht['q1_diff']:+d})")
    print(f"  Q2: {home['abbr']} {ht['q2_home_pts']} - {away['abbr']} {ht['q2_away_pts']} "
          f"(diff: {ht['q2_diff']:+d})")
    print(f"  Momentum shift: {ht['momentum_shift']:+d} "
          f"({'home' if ht['momentum_shift'] > 0 else 'away'} gaining)")
    print(f"\nFinal    : {home['abbr']} {final['home_final_pts']} - "
          f"{away['abbr']} {final['away_final_pts']} "
          f"({'HOME WIN' if final['home_win'] else 'AWAY WIN'}, "
          f"margin: {final['final_margin']:+d})")
    print(f"{'─'*60}\n")
