"""
Ajuste por QB starter.
Idea central: separar la EPA ofensiva de un equipo en dos partes:
  - EPA de carrera (mas atribuible a linea ofensiva/RBs -> "pegajoso" al equipo)
  - EPA de pase por dropback (mas atribuible al QB -> debe viajar CON el jugador, no con el equipo)
Asi, cuando un equipo cambia de QB (via free agency, draft, lesion), el modelo
usa el historial del QB nuevo, no el promedio viejo del equipo.
"""
import nflreadpy as nfl
import polars as pl
import numpy as np

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
def normalize_team(t):
    return TEAM_MAP.get(t, t)

def load_pbp_clean(seasons):
    pbp = nfl.load_pbp(seasons=seasons)
    pbp = pbp.with_columns([
        pl.col("posteam").map_elements(normalize_team, return_dtype=pl.Utf8, skip_nulls=True).alias("posteam"),
    ])
    return pbp

def team_rush_epa(pbp: pl.DataFrame):
    """EPA de carrera por equipo (proxy de o-linea/RBs, mas 'pegajoso' que el pase)."""
    rush = pbp.filter((pl.col("play_type") == "run") & pl.col("epa").is_not_null())
    return rush.group_by(["season", "posteam"]).agg(pl.col("epa").mean().alias("rush_epa"))

def qb_career_value(pbp: pl.DataFrame, decay_per_season=0.6, min_dropbacks_season=100):
    """
    EPA por dropback de cada QB, ponderado por temporada (decae con el tiempo,
    para que un buen año hace 5 temporadas pese poco hoy).
    Devuelve una tabla season x qb con su 'valor acumulado hasta esa temporada' (sin leakage).
    """
    db = pbp.filter((pl.col("qb_dropback") == 1) & pl.col("qb_epa").is_not_null() &
                    pl.col("passer_player_id").is_not_null())

    season_qb = db.group_by(["season", "passer_player_id", "passer_player_name"]).agg([
        pl.col("qb_epa").mean().alias("epa_per_dropback"),
        pl.len().alias("n_dropbacks")
    ])

    seasons = sorted(season_qb["season"].unique().to_list())
    results = []

    for i, s in enumerate(seasons):
        for qb_id in season_qb.filter(pl.col("season") == s)["passer_player_id"].unique().to_list():
            qb_hist = season_qb.filter(
                (pl.col("passer_player_id") == qb_id) & (pl.col("season") <= s)
            ).sort("season")

            if qb_hist.height == 0:
                continue

            weights = np.array([decay_per_season ** (s - row["season"]) for row in qb_hist.iter_rows(named=True)])
            values = qb_hist["epa_per_dropback"].to_numpy()
            n = qb_hist["n_dropbacks"].to_numpy()
            # Pondera tambien por volumen de dropbacks (una temporada de 20 pases pesa menos que una de 500)
            sample_weight = np.minimum(n / min_dropbacks_season, 1.0)
            final_weights = weights * sample_weight

            if final_weights.sum() == 0:
                continue

            weighted_value = np.average(values, weights=final_weights)
            name = qb_hist[-1]["passer_player_name"][0]
            results.append({"season": s, "qb_id": qb_id, "qb_name": name,
                             "qb_value_epa_dropback": weighted_value,
                             "career_dropbacks_to_date": int(n.sum())})

    return pl.DataFrame(results)

def primary_starter_per_team_season(pbp: pl.DataFrame):
    """Quien tuvo mas dropbacks por equipo-temporada = el 'starter primario' esa temporada."""
    db = pbp.filter((pl.col("qb_dropback") == 1) & pl.col("passer_player_id").is_not_null())
    counts = db.group_by(["season", "posteam", "passer_player_id", "passer_player_name"]).agg(
        pl.len().alias("dropbacks")
    )
    return counts.sort(["season", "posteam", "dropbacks"], descending=[False, False, True]) \
                 .group_by(["season", "posteam"], maintain_order=True).first()


if __name__ == "__main__":
    pbp = load_pbp_clean(range(2018, 2025))
    qb_values = qb_career_value(pbp)
    starters = primary_starter_per_team_season(pbp)

    league_avg = qb_values.filter(pl.col("season") == 2024)["qb_value_epa_dropback"].mean()
    print(f"Promedio de liga (valor QB, 2024): {league_avg:.4f}\n")

    # Ejemplo real: casos de cambio de starter que deberian mostrar el efecto
    for team, season in [("MIN", 2024), ("WAS", 2024), ("MIN", 2023)]:
        starter_row = starters.filter((pl.col("season") == season) & (pl.col("posteam") == team))
        if starter_row.height == 0:
            continue
        qb_id = starter_row["passer_player_id"][0]
        qb_name = starter_row["passer_player_name"][0]
        val = qb_values.filter((pl.col("season") == season) & (pl.col("qb_id") == qb_id))
        v = val["qb_value_epa_dropback"][0] if val.height > 0 else None
        print(f"{team} {season}: starter={qb_name}, valor QB (EPA/dropback ponderado)={v}")


def draft_round_bucket(round_num):
    """Agrupamos en cubetas mas anchas para evitar ruido de muestra chica por ronda individual."""
    if round_num == 1:
        return "R1"
    elif round_num in (2, 3):
        return "R2_R3"
    else:
        return "R4_PLUS"

UDFA_BUCKET = "UDFA"  # no draftead - bucket mas bajo por default

def build_rookie_prior_baseline(qb_values: pl.DataFrame, draft_picks: pl.DataFrame):
    """
    Baseline empirico: valor promedio de un QB novato segun donde fue draftado,
    usando cubetas anchas (R1 / R2-R3 / R4+) para tener muestra suficiente.
    """
    qb_picks = draft_picks.filter(pl.col("position") == "QB").select(
        ["season", "round", "gsis_id"]
    ).with_columns(
        pl.col("round").map_elements(draft_round_bucket, return_dtype=pl.Utf8).alias("bucket")
    )

    rookie_year = qb_values.join(
        qb_picks, left_on=["season", "qb_id"], right_on=["season", "gsis_id"], how="inner"
    )

    baseline = rookie_year.group_by("bucket").agg([
        pl.col("qb_value_epa_dropback").mean().alias("baseline_value"),
        pl.len().alias("n_qbs")
    ])
    return baseline

def rookie_qb_value_with_shrinkage(observed_value, observed_dropbacks, bucket_baseline,
                                     full_shrinkage_dropbacks=150):
    """
    Bayesian shrinkage simple: con pocos dropbacks en la NFL, confiamos mas en el
    baseline de su bucket de draft. Con muchos dropbacks propios, confiamos mas en
    su propio desempeño observado.
    Si observed_value es None (0 dropbacks, ej. antes de su primer arranque), usa 100% el baseline.
    """
    if observed_value is None or observed_dropbacks == 0:
        return bucket_baseline

    weight_own = min(observed_dropbacks / full_shrinkage_dropbacks, 1.0)
    return weight_own * observed_value + (1 - weight_own) * bucket_baseline


if __name__ == "__main__":
    pbp = load_pbp_clean(range(2015, 2025))
    qb_values = qb_career_value(pbp)
    picks = nfl.load_draft_picks()

    baseline = build_rookie_prior_baseline(qb_values, picks)
    print("--- Baseline por cubeta de draft (rookie season) ---")
    print(baseline)

    # Ejemplo: un rookie con muy pocos dropbacks propios (simulando semana 2 de su carrera)
    r1_baseline = baseline.filter(pl.col("bucket") == "R1")["baseline_value"][0]
    print(f"\nEjemplo: rookie R1 con 0 dropbacks propios -> valor asignado: {r1_baseline:.4f}")
    print(f"Ejemplo: rookie R1 con 60 dropbacks propios y epa observado=0.20 -> "
          f"{rookie_qb_value_with_shrinkage(0.20, 60, r1_baseline):.4f}")