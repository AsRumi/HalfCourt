"""
prompt_builder.py

Build LLM prompt from a feature vector, model prediction, and game context.
"""

SYSTEM_PROMPT = """You are a professional NBA broadcast analyst writing a halftime report.
HARD CONSTRAINT: The predicted winner and margin below are facts. Do not contradict them or express uncertainty about the outcome.
GROUNDING RULE: Only reference statistics explicitly provided in the halftime data block. Do not invent or recall statistics from your training data.
STYLE: Confident, specific, player-focused. No generic filler phrases."""


def _game_context_block(game_context: dict) -> str:
    playoff_label = "Yes" if game_context.get("is_playoffs") else "No"
    return (
        f"GAME CONTEXT\n"
        f"Home Team : {game_context['home_team']}\n"
        f"Away Team : {game_context['away_team']}\n"
        f"Date      : {game_context['game_date']}\n"
        f"Playoffs  : {playoff_label}"
    )


def _stats_table(feature_vector: dict, game_context: dict) -> str:
    home = game_context["home_team"]
    away = game_context["away_team"]

    ht_home = game_context["ht_home_pts"]
    ht_away = game_context["ht_away_pts"]
    q1_home = game_context["q1_home_pts"]
    q1_away = game_context["q1_away_pts"]
    q2_home = game_context["q2_home_pts"]
    q2_away = game_context["q2_away_pts"]

    def row(label, home_val, away_val, fmt=".0f"):
        return f"{label:<28} {home_val:>16{fmt}} {away_val:>16{fmt}}"

    lines = [
        "HALFTIME STATS",
        f"{'Stat':<28} {'Home (' + home + ')':>16} {'Away (' + away + ')':>16}",
        "-" * 62,
        row("Halftime Score", ht_home, ht_away),
        row("Q1 Score", q1_home, q1_away),
        row("Q2 Score", q2_home, q2_away),
        row("Momentum Shift", feature_vector["momentum_shift"], "", fmt=""),
        row("FG%", feature_vector["home_fg_pct"], feature_vector["away_fg_pct"], fmt=".3f"),
        row("3P%", feature_vector["home_3p_pct"], feature_vector["away_3p_pct"], fmt=".3f"),
        row("FT%", feature_vector["home_ft_pct"], feature_vector["away_ft_pct"], fmt=".3f"),
        row("eFG%", feature_vector["home_efg_pct"], feature_vector["away_efg_pct"], fmt=".3f"),
        row("TS%", feature_vector["home_ts_pct"], feature_vector["away_ts_pct"], fmt=".3f"),
        row("Turnovers", feature_vector["home_to"], feature_vector["away_to"]),
        row("AST/TO Ratio", feature_vector["home_ast_to_ratio"], feature_vector["away_ast_to_ratio"], fmt=".2f"),
        row("OREB%", feature_vector["home_oreb_pct"], feature_vector["away_oreb_pct"], fmt=".3f"),
        row("Rebound Diff (home-away)", feature_vector["reb_diff"], "", fmt=""),
        row("Bench Points", feature_vector["home_bench_pts"], feature_vector["away_bench_pts"]),
        row("Bench Ratio", feature_vector["home_bench_ratio"], feature_vector["away_bench_ratio"], fmt=".3f"),
        row("Starters w/ 3+ Fouls", feature_vector["home_starters_3foul"], feature_vector["away_starters_3foul"]),
        row("Foul Diff (home-away)", feature_vector["foul_diff"], "", fmt=""),
        row(f"Star PTS ({game_context['home_star_name']} / {game_context['away_star_name']})", feature_vector["home_star1_pts"], feature_vector["away_star1_pts"]),
        row("Star Assists", feature_vector["home_star1_ast"], feature_vector["away_star1_ast"]),
        row("Star Foul Flag (3+ PF)", feature_vector["home_star_foul_flag"], feature_vector["away_star_foul_flag"]),
        row("Rest Days", feature_vector["home_rest_days"], feature_vector["away_rest_days"]),
        row("Back-to-Back", feature_vector["home_back2back"], feature_vector["away_back2back"]),
    ]
    return "\n".join(lines)


def _prediction_block(prediction: dict, game_context: dict) -> str:
    winner_name = (
        game_context["home_team"] if prediction["predicted_winner"] == "home"
        else game_context["away_team"]
    )
    prob_pct = round(prediction["win_probability"] * 100, 1)
    raw_margin = prediction["predicted_margin"]
    margin_line = (
        f"Predicted Margin   : {winner_name} by {abs(raw_margin)} points"
        if raw_margin is not None
        else "Predicted Margin   : N/A (Logistic Regression does not predict margin)"
    )

    home_star = game_context.get("home_star_name", "")
    away_star = game_context.get("away_star_name", "")

    feature_lines = []
    for f in prediction["top_features"]:
        direction = (
            game_context["home_team"] if f["direction"] == "favors_home"
            else game_context["away_team"]
        )
        feature_lines.append(f"  - {f['feature']} (SHAP: {f['value']:+.4f}) -> favors {direction}")

    star_context = (
        f"\nKey Players: {game_context['home_team']} star: {home_star} | "
        f"{game_context['away_team']} star: {away_star}"
    )

    features_str = "\n".join(feature_lines)

    return (
        f"MODEL PREDICTION\n"
        f"Predicted Winner   : {winner_name}\n"
        f"Win Probability    : {prob_pct}% (home team)\n"
        f"{margin_line}\n"
        f"Confidence Tier    : {prediction['confidence_tier'].upper()}\n"
        f"{star_context}\n"
        f"\nTop Driving Features:\n{features_str}"
    )


def _output_format_instruction() -> str:
    return (
        "OUTPUT FORMAT\n"
        "Write your halftime report in exactly four sections with these headers:\n\n"
        "1. SCORE & MOMENTUM\n"
        "2. KEY PLAYER ANALYSIS\n"
        "3. X-FACTORS / FOUL TROUBLE\n"
        "4. SECOND-HALF OUTLOOK\n\n"
        "Each section should be 2-4 sentences. Do not add any other sections."
    )


def build_prompt(feature_vector: dict, prediction: dict, game_context: dict) -> str:
    """
    Assemble the full LLM prompt from feature vector, prediction, and game context.

    Args:
        feature_vector: flat feature dict from feature_engineer.py
        prediction:     output dict from predictor.predict()
        game_context:   dict including home_team, away_team, game_date, is_playoffs,
                        ht_home_pts, ht_away_pts, q1_home_pts, q1_away_pts,
                        q2_home_pts, q2_away_pts

    Returns:
        A dict with "system" and "user" keys, both strings, ready for LLM submission.
    """
    user_message = "\n\n".join([
        _game_context_block(game_context),
        _stats_table(feature_vector, game_context),
        _prediction_block(prediction, game_context),
        _output_format_instruction(),
    ])

    return {"system": SYSTEM_PROMPT, "user": user_message}
