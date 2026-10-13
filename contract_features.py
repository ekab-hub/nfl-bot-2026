"""
Tres variables nuevas basadas en contratos (load_contracts):
  1. mega_contract_count: cuantos jugadores de "estrella" (cap_percent alto) tiene el equipo
  2. starting_lineup_draft_capital: capital de draft promedio de los titulares reales
  3. offseason_spending: cuanto comprometio el equipo en contratos NUEVOS ese año

Todas construidas por equipo-temporada, listas para probar significancia
contra el cambio de wins del año siguiente -- mismo estandar que luck,
qb_delta e injury_burden.
"""
import nflreadpy as nfl
import polars as pl
import numpy as np

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
def normalize_team(t):
    return TEAM_MAP.get(t, t)

MEGA_CONTRACT_THRESHOLD = 0.08  # 8% del cap total -- umbral razonable de "contrato de estrella"


def load_player_season_cap():
    """
    Explota season_history (una fila por jugador-temporada), deduplicando
    porque el mismo jugador aparece repetido por cada contrato distinto
    que firmo en su carrera (season_history se repite en cada fila).
    """
    c = nfl.load_contracts()
    exploded = c.select(["player", "gsis_id", "season_history"]).explode("season_history").unnest("season_history")
    exploded = exploded.filter(pl.col("year") != "Total")  # excluye la fila resumen de carrera completa
    exploded = exploded.with_columns(pl.col("year").cast(pl.Int64))
    exploded = exploded.unique(subset=["player", "gsis_id", "year", "team"])
    exploded = exploded.with_columns(
        pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8, skip_nulls=True).alias("team")
    )
    return exploded


def load_player_signed_years():
    """Años distintos en que cada jugador firmo un contrato nuevo (puede tener varios en su carrera)."""
    c = nfl.load_contracts()
    return c.select(["gsis_id", "year_signed"]).unique().filter(pl.col("year_signed") > 0)


def build_mega_contract_count(seasons):
    player_cap = load_player_season_cap()
    player_cap = player_cap.filter(pl.col("year").is_in(list(seasons)))
    counts = player_cap.filter(pl.col("cap_percent") >= MEGA_CONTRACT_THRESHOLD).group_by(["year", "team"]).agg(
        pl.len().alias("mega_contract_count")
    )
    return counts.rename({"year": "season"})


def build_offseason_spending(seasons):
    player_cap = load_player_season_cap()
    signed_years = load_player_signed_years()

    # Marca si esa temporada especifica de ese jugador corresponde al año
    # en que firmo ALGUNO de sus contratos (nuevo compromiso financiero)
    tagged = player_cap.join(signed_years, on="gsis_id", how="inner")
    new_deals = tagged.filter(pl.col("year") == pl.col("year_signed"))
    new_deals = new_deals.filter(pl.col("year").is_in(list(seasons)))

    spending = new_deals.group_by(["year", "team"]).agg(
        pl.col("guaranteed_salary").sum().alias("offseason_spending")
    )
    return spending.rename({"year": "season"})


def build_starting_lineup_draft_capital(seasons):
    """
    Capital de draft promedio de los titulares reales (por snaps), ponderado
    por participacion. Ronda mas baja (mas temprana) = mejor capital, se
    invierte la escala para que "mayor numero = mejor capital invertido".
    """
    rows = []
    picks = nfl.load_draft_picks().select(["pfr_player_id", "round"]).unique(subset=["pfr_player_id"])

    for season in seasons:
        snaps = nfl.load_snap_counts(seasons=[season])
        snaps = snaps.with_columns(
            pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8, skip_nulls=True).alias("team"),
            pl.max_horizontal(["offense_pct", "defense_pct"]).alias("snap_pct"),
        )
        player_snaps = snaps.group_by(["team", "player", "pfr_player_id"]).agg(
            pl.col("snap_pct").mean().alias("avg_snap_pct")
        )
        # gsis_id no siempre viene en snap_counts -- se cruza por pfr_player_id si esta disponible,
        # si no hay cruce valido para un jugador simplemente no aporta capital de draft ese caso
        merged = player_snaps.join(picks, on="pfr_player_id", how="left")
        merged = merged.with_columns(
            (33 - pl.col("round")).fill_null(0).alias("draft_capital_score")  # ronda 1 -> 32, sin draftear -> 0
        )
        team_capital = merged.group_by("team").agg(
            (pl.col("draft_capital_score") * pl.col("avg_snap_pct")).sum().alias("weighted_capital"),
            pl.col("avg_snap_pct").sum().alias("total_snap_weight")
        )
        team_capital = team_capital.with_columns(
            (pl.col("weighted_capital") / pl.col("total_snap_weight")).alias("starting_lineup_draft_capital")
        )
        team_capital = team_capital.with_columns(pl.lit(season).alias("season"))
        rows.append(team_capital.select(["season", "team", "starting_lineup_draft_capital"]))

    return pl.concat(rows)


if __name__ == "__main__":
    seasons = range(2018, 2026)
    mega = build_mega_contract_count(seasons)
    spending = build_offseason_spending(seasons)
    capital = build_starting_lineup_draft_capital(seasons)

    print("--- Mega-contratos 2024, top 10 ---")
    print(mega.filter(pl.col("season") == 2024).sort("mega_contract_count", descending=True).head(10))

    print("\n--- Gasto en offseason 2024, top 10 (millones USD, guaranteed) ---")
    print(spending.filter(pl.col("season") == 2024).sort("offseason_spending", descending=True).head(10))

    print("\n--- Capital de draft del starting lineup 2024, top 10 ---")
    print(capital.filter(pl.col("season") == 2024).sort("starting_lineup_draft_capital", descending=True).head(10))