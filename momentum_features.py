"""
Dos variables de momentum, por partido:
  A. win_streak: victorias consecutivas entrando al partido
  B. surprise_momentum: cuanto por encima/debajo de lo esperado por Elo ha
     rendido el equipo en sus ultimos 3 partidos (ganarle a un favorito
     pesa mas que ganarle a un equipo malo)

Ambas se resetean cada temporada (racha de la temporada pasada no cuenta
como "momentum" al arrancar una nueva, hay demasiado tiempo/cambio de por medio).
"""
import polars as pl
from motor_elo_v1 import EloEngine, load_games

SURPRISE_WINDOW = 3


def build_momentum_features(seasons, home_advantage=35, k=20, regression_to_mean=0.40):
    games = load_games(seasons)
    engine = EloEngine(k=k, home_advantage=home_advantage, regression_to_mean=regression_to_mean)

    team_history = {}  # team -> lista de {"result": 1/0/0.5, "surprise": float}, se resetea cada temporada
    rows = []
    current_season = None

    for row in games.iter_rows(named=True):
        if current_season is not None and row["season"] != current_season:
            engine.new_season_regression()
            team_history = {}  # reset de momentum entre temporadas
        current_season = row["season"]

        home, away = row["home_team"], row["away_team"]
        home_hist = team_history.get(home, [])
        away_hist = team_history.get(away, [])

        def win_streak(hist):
            streak = 0
            for h in reversed(hist):
                if h["result"] == 1:
                    streak += 1
                else:
                    break
            return streak

        def surprise_momentum(hist):
            recent = hist[-SURPRISE_WINDOW:]
            return sum(h["surprise"] for h in recent) if recent else 0.0

        r_home = engine.get_rating(home) + engine.home_advantage
        r_away = engine.get_rating(away)
        elo_prob_home = engine.expected_win_prob(r_home, r_away)

        rows.append({
            "season": row["season"], "week": row["week"],
            "home_team": home, "away_team": away,
            "elo_prob_home": elo_prob_home,
            "win_streak_home": win_streak(home_hist), "win_streak_away": win_streak(away_hist),
            "momentum_home": surprise_momentum(home_hist), "momentum_away": surprise_momentum(away_hist),
            "home_score": row["home_score"], "away_score": row["away_score"],
        })

        prob_home_pre = engine.update_game(home, away, row["home_score"], row["away_score"])
        home_result = 1.0 if row["home_score"] > row["away_score"] else (0.0 if row["home_score"] < row["away_score"] else 0.5)
        away_result = 1.0 - home_result

        team_history.setdefault(home, []).append({"result": home_result, "surprise": home_result - prob_home_pre})
        team_history.setdefault(away, []).append({"result": away_result, "surprise": away_result - (1 - prob_home_pre)})

    return pl.DataFrame(rows)


if __name__ == "__main__":
    df = build_momentum_features(range(2015, 2026))
    df = df.with_columns([
        (pl.col("win_streak_home") - pl.col("win_streak_away")).alias("streak_diff"),
        (pl.col("momentum_home") - pl.col("momentum_away")).alias("momentum_diff"),
    ])
    print(f"Total partidos: {df.height}")
    print(df.select(["season", "week", "home_team", "away_team", "win_streak_home",
                      "win_streak_away", "momentum_home", "momentum_away"]).tail(10))
    df.write_parquet("momentum_features.parquet")
    print("\nGuardado en cache: momentum_features.parquet")