"""
QBs fuera POR EL RESTO DE LA TEMPORADA 2026 (estilo starters_2026.py), para el simulador
de temporada en modo en vivo (use_played=True).

QB_OUT_FOR_SEASON = {"EQUIPO": ("QB lesionado", "QB reemplazo")}, nombres como en el roster
de nflreadpy. Vacio por defecto -> el simulador queda exactamente igual.

La caida de valor (qb_value_epa_dropback del pipeline, valor antes de la temporada) se
convierte a Elo con el b calibrado contra el mercado (experimentos/REPORTE_QB.md):
  puntos de spread = B_SPREAD_PER_QB_VALUE * (valor_lesionado - valor_reemplazo)
  puntos de Elo    = -25 * puntos de spread   (negativo = penalizacion)
"""
import nflreadpy as nfl
import polars as pl
from qb_adjustment import load_pbp_clean, qb_career_value, build_rookie_prior_baseline

QB_OUT_FOR_SEASON = {
    # "KC": ("Patrick Mahomes", "Justin Fields"),   # ejemplo
    "NYG": ('Jaxson Dart', 'Jameis Winston')
}

B_SPREAD_PER_QB_VALUE = 8.40   # spec (ii) 2010-2022, SE 0.79 (experimentos/qb_mercado.py)
ELO_PER_POINT = 25.0
VALUE_HISTORY_START = 2015


def qb_elo_adjustments(season=2026, qb_out=None, verbose=True):
    """dict equipo -> puntos de Elo para el simulador. Imprime cada ajuste aplicado."""
    qb_out = QB_OUT_FOR_SEASON if qb_out is None else qb_out
    if not qb_out:
        return {}

    pbp = load_pbp_clean(range(VALUE_HISTORY_START, season))
    qb_values = qb_career_value(pbp)
    picks = nfl.load_draft_picks()
    rookie_baseline = build_rookie_prior_baseline(qb_values, picks)
    rosters = nfl.load_rosters(seasons=[season]).filter(pl.col("position") == "QB")

    def resolve(team, name):
        m = rosters.filter((pl.col("team") == team) & (pl.col("full_name") == name))
        if m.height == 0:
            m = rosters.filter(pl.col("full_name") == name)
        if m.height == 0:
            raise ValueError(f"QB '{name}' ({team}) no encontrado en el roster {season}")
        return m["gsis_id"][0]

    def value(qb_id):
        hist = qb_values.filter((pl.col("qb_id") == qb_id) & (pl.col("season") < season))
        if hist.height > 0:
            return hist.sort("season", descending=True)["qb_value_epa_dropback"][0]
        pick = picks.filter((pl.col("gsis_id") == qb_id) & (pl.col("position") == "QB"))
        rnd = pick["round"][0] if pick.height > 0 else None
        bucket = "R1" if rnd == 1 else ("R2_R3" if rnd in (2, 3) else "R4_PLUS")
        return rookie_baseline.filter(pl.col("bucket") == bucket)["baseline_value"][0]

    adjustments = {}
    for team, (out_name, repl_name) in qb_out.items():
        v_out = value(resolve(team, out_name))
        v_rep = value(resolve(team, repl_name))
        spread_pts = B_SPREAD_PER_QB_VALUE * (v_out - v_rep)
        adjustments[team] = -ELO_PER_POINT * spread_pts
        if verbose:
            print(f"(ajuste QB) {team}: {out_name} ({v_out:+.3f}) -> {repl_name} ({v_rep:+.3f}): "
                  f"{-spread_pts:+.2f} pts de margen = {adjustments[team]:+.1f} Elo")
    return adjustments
