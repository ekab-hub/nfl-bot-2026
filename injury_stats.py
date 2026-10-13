"""
Construye una metrica de 'carga de lesiones ponderada' por equipo-temporada:
para cada jugador marcado 'Out' en el reporte de lesiones, se pesa por su
% de snaps promedio (importancia real en el equipo, medida en las semanas
que SI jugo). Es el mismo concepto que 'Adjusted Games Lost' (AGL) usado
en la comunidad de analitica NFL (Football Outsiders), construido aqui con
evidencia propia, no asumido.
"""
import nflreadpy as nfl
import polars as pl

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
def normalize_team(t):
    return TEAM_MAP.get(t, t)


def build_injury_burden(seasons):
    rows = []
    for season in seasons:
        injuries = nfl.load_injuries(seasons=[season])
        injuries = injuries.filter(pl.col("report_status") == "Out")
        injuries = injuries.with_columns([
            pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("team"),
            pl.col("season").cast(pl.Int64).alias("season"),
        ])

        snaps = nfl.load_snap_counts(seasons=[season])
        snaps = snaps.with_columns([
            pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("team"),
            pl.col("season").cast(pl.Int64).alias("season"),
        ])
        snaps = snaps.with_columns(
            pl.max_horizontal(["offense_pct", "defense_pct"]).alias("snap_pct")
        )
        player_importance = snaps.group_by(["season", "team", "player"]).agg(
            pl.col("snap_pct").mean().alias("avg_snap_pct")
        )

        weighted = injuries.join(
            player_importance, left_on=["season", "team", "full_name"],
            right_on=["season", "team", "player"], how="left"
        )
        weighted = weighted.with_columns(pl.col("avg_snap_pct").fill_null(0.0))

        burden = weighted.group_by(["season", "team"]).agg(
            pl.col("avg_snap_pct").sum().alias("injury_burden")
        )
        rows.append(burden)

    return pl.concat(rows)


if __name__ == "__main__":
    burden = build_injury_burden(range(2018, 2026))
    print("--- Equipos con mayor carga de lesiones en 2024 ---")
    print(burden.filter(pl.col("season") == 2024).sort("injury_burden", descending=True).head(10))
    print(f"\nTotal filas: {burden.height}")
    print(burden["injury_burden"].describe())