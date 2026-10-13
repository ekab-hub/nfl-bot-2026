"""
Chequeo de sesgo "campeon defensor": dos preguntas distintas.
  1. Realidad: que tan seguido el campeon de la temporada N-1 vuelve a ganar
     en N (o al menos sigue siendo top contendiente)? -- linea base real,
     independiente de nuestro modelo.
  2. Nuestro modelo: para cada temporada historica, el campeon defensor
     aparece como el FAVORITO del propio modelo (preseason, antes de ver
     un partido)? Si el patron es demasiado fuerte y constante, podria
     indicar que el modelo esta sobre-pesando el Elo heredado del campeon
     en vez de evaluar la temporada nueva con criterio propio.
"""
import polars as pl
import nflreadpy as nfl
from season_sim_core import get_preseason_ratings, get_season_schedule, simulate_one_season, get_all_division_winners, normalize_team
from division_champions import run_division_monte_carlo
import sys
sys.path.insert(0, ".")
from superbowl_champions import run_superbowl_monte_carlo

TEST_SEASONS = range(2006, 2026)  # necesita el campeon de N-1, por eso arranca un año despues del historico base


def get_actual_champion(season):
    df = nfl.load_schedules(seasons=[season])
    sb = df.filter(pl.col("game_type") == "SB")
    if sb.height == 0:
        return None
    row = sb.row(0, named=True)
    winner = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
    return normalize_team(winner)


if __name__ == "__main__":
    rows = []

    for season in TEST_SEASONS:
        prev_champion = get_actual_champion(season - 1)
        actual_champion = get_actual_champion(season)
        if prev_champion is None or actual_champion is None:
            continue

        print(f"Simulando temporada {season} (campeon defensor: {prev_champion})...")
        results = run_superbowl_monte_carlo(season, n_sims=10000)

        results_sorted = results.sort("sb_champion_prob", descending=True)
        model_favorite = results_sorted["team"][0]

        prev_champ_row = results.filter(pl.col("team") == prev_champion)
        prev_champ_prob = prev_champ_row["sb_champion_prob"][0] if prev_champ_row.height > 0 else None
        prev_champ_rank = results_sorted.with_row_index("rank").filter(pl.col("team") == prev_champion)["rank"][0] + 1

        rows.append({
            "season": season,
            "campeon_defensor": prev_champion,
            "favorito_del_modelo": model_favorite,
            "modelo_repite_favorito": model_favorite == prev_champion,
            "prob_campeon_defensor": round(prev_champ_prob, 4) if prev_champ_prob else None,
            "ranking_campeon_defensor": int(prev_champ_rank),
            "campeon_real_ese_año": actual_champion,
            "campeon_defensor_repitio_en_realidad": actual_champion == prev_champion,
        })

    df = pl.DataFrame(rows)
    with pl.Config(tbl_rows=25, tbl_cols=10):
        print(df)

    print(f"\n=== Pregunta 1: realidad ===")
    real_repeat_rate = df["campeon_defensor_repitio_en_realidad"].mean()
    print(f"El campeon defensor volvio a ganar el Super Bowl: {real_repeat_rate*100:.1f}% de las veces ({df['campeon_defensor_repitio_en_realidad'].sum()} de {df.height} temporadas)")

    print(f"\n=== Pregunta 2: nuestro modelo ===")
    model_repeat_rate = df["modelo_repite_favorito"].mean()
    print(f"El MODELO puso al campeon defensor como favorito absoluto: {model_repeat_rate*100:.1f}% de las veces ({df['modelo_repite_favorito'].sum()} de {df.height} temporadas)")
    print(f"Ranking promedio del campeon defensor en la proyeccion del modelo: {df['ranking_campeon_defensor'].mean():.1f} de 32")
    print(f"Probabilidad promedio que el modelo le dio al campeon defensor: {df['prob_campeon_defensor'].mean()*100:.2f}%")