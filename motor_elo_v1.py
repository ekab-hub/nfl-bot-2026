"""
Motor Elo para NFL - v1
Incluye: reconciliacion de franquicias relocalizadas, home advantage,
multiplicador por margen de victoria (MOV), y regresion a la media entre temporadas.
"""
import nflreadpy as nfl
import polars as pl
import math

# --- Reconciliacion de nombres (franquicias relocalizadas) ---
TEAM_MAP = {
    "OAK": "LV",   # Raiders: Oakland -> Las Vegas (2020)
    "SD": "LAC",   # Chargers: San Diego -> LA (2017)
    "STL": "LA",   # Rams: St. Louis -> LA (2016)
}

def normalize_team(team: str) -> str:
    return TEAM_MAP.get(team, team)


def load_games(seasons):
    df = nfl.load_schedules(seasons=seasons)
    # Solo partidos con resultado (regular season + playoffs ya jugados)
    df = df.filter(pl.col("home_score").is_not_null())
    df = df.with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])
    return df.sort(["season", "week", "gameday"])


class EloEngine:
    def __init__(self, k=20, home_advantage=35, base_rating=1500,
                 regression_to_mean=0.4):
        """
        k: velocidad de ajuste por partido
        home_advantage: puntos Elo equivalentes a jugar de local
        regression_to_mean: fraccion que se revierte al promedio de liga entre temporadas
                            (1/3 es un punto de partida estandar en modelos publicos tipo 538)
        """
        self.k = k
        self.home_advantage = home_advantage
        self.base_rating = base_rating
        self.regression_to_mean = regression_to_mean
        self.ratings = {}

    def get_rating(self, team):
        return self.ratings.get(team, self.base_rating)

    def expected_win_prob(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def mov_multiplier(self, margin, elo_diff):
        # Formula estilo 538: partidos mas lopsided mueven mas el rating,
        # pero se amortigua si el favorito ya era muy favorito (evita overreaction)
        return math.log(abs(margin) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def new_season_regression(self):
        league_avg = self.base_rating
        for team in list(self.ratings.keys()):
            r = self.ratings[team]
            self.ratings[team] = r + self.regression_to_mean * (league_avg - r)

    def update_game(self, home_team, away_team, home_score, away_score):
        r_home = self.get_rating(home_team) + self.home_advantage
        r_away = self.get_rating(away_team)

        exp_home = self.expected_win_prob(r_home, r_away)
        margin = home_score - away_score

        if margin > 0:
            actual_home = 1.0
        elif margin < 0:
            actual_home = 0.0
        else:
            actual_home = 0.5

        elo_diff = r_home - r_away
        mult = self.mov_multiplier(margin if margin != 0 else 1, elo_diff)

        shift = self.k * mult * (actual_home - exp_home)

        self.ratings[home_team] = self.get_rating(home_team) + shift
        self.ratings[away_team] = self.get_rating(away_team) - shift

        return exp_home  # probabilidad pre-partido de que ganara el local


def run_elo_over_seasons(seasons, k=20, home_advantage=35, regression_to_mean=0.40):
    games = load_games(seasons)
    engine = EloEngine(k=k, home_advantage=home_advantage, regression_to_mean=regression_to_mean)

    predictions = []
    current_season = None

    for row in games.iter_rows(named=True):
        if current_season is not None and row["season"] != current_season:
            engine.new_season_regression()
        current_season = row["season"]

        pre_elo_home = engine.get_rating(row["home_team"])
        pre_elo_away = engine.get_rating(row["away_team"])

        prob_home = engine.update_game(
            row["home_team"], row["away_team"],
            row["home_score"], row["away_score"]
        )

        predictions.append({
            "season": row["season"], "week": row["week"], "game_type": row["game_type"],
            "home_team": row["home_team"], "away_team": row["away_team"],
            "home_score": row["home_score"], "away_score": row["away_score"],
            "pre_elo_home": pre_elo_home, "pre_elo_away": pre_elo_away,
            "model_prob_home": prob_home,
            "spread_line": row["spread_line"],
            "home_moneyline": row["home_moneyline"],
        })

    return pl.DataFrame(predictions, infer_schema_length=None), engine