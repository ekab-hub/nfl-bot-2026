"""
Diagnostico de division: descompone de donde sale la probabilidad de titulo
de un equipo (rating base, ajuste luck, resultados jugados, calendario
restante) y la compara con el mercado.
"""
import numpy as np
import polars as pl
from motor_elo_v1 import run_elo_over_seasons
from season_sim_core import (get_preseason_ratings, get_season_schedule, simulate_one_season,
                              resolve_division_winner, DIVISIONS, TEAM_TO_DIVISION)

SEASON = 2026
TEAM = "NYG"
N_SIMS = 10000
MARKET_AMERICAN = {"NYG": 600}  # agrega PHI, WAS, DAL si tienes sus momios


def american_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def base_ratings_without_luck(season):
    _, engine = run_elo_over_seasons(list(range(2015, season)))
    engine.new_season_regression()
    return dict(engine.ratings)


def division_title_probs(ratings, schedule, division_teams, n_sims):
    wins = {t: 0 for t in division_teams}
    for _ in range(n_sims):
        records, _ = simulate_one_season(ratings, schedule)
        wins[resolve_division_winner(records, division_teams)] += 1
    return {t: w / n_sims for t, w in wins.items()}


if __name__ == "__main__":
    division = TEAM_TO_DIVISION[TEAM]
    teams = DIVISIONS[division]

    schedule_full = get_season_schedule(SEASON)
    schedule_played = get_season_schedule(SEASON, include_played_results=True)
    played = [g for g in schedule_played if g.get("home_score") is not None]
    remaining = [g for g in schedule_played if g.get("home_score") is None]
    print(f"Partidos jugados: {len(played)} | restantes: {len(remaining)}\n")

    base = base_ratings_without_luck(SEASON)
    pre = get_preseason_ratings(SEASON)
    records_now, elo_now = simulate_one_season(pre, played)

    rows = []
    for t in teams:
        r = records_now[t]
        opp_elo = [elo_now[g["away_team"] if g["home_team"] == t else g["home_team"]]
                   for g in remaining if t in (g["home_team"], g["away_team"])]
        rows.append({
            "team": t,
            "record": f"{r['w']}-{r['l']}" + (f"-{r['t']}" if r["t"] else ""),
            "pf-pa": f"{r['pf']}-{r['pa']}",
            "elo_base_sin_luck": round(base[t], 1),
            "ajuste_luck": round(pre[t] - base[t], 1),
            "elo_preseason": round(pre[t], 1),
            "elo_actual": round(elo_now[t], 1),
            "elo_rivales_restantes": round(float(np.mean(opp_elo)), 1),
        })
    print(f"=== {division} ===")
    print(pl.DataFrame(rows))

    sc_a = division_title_probs(pre, schedule_full, teams, N_SIMS)
    sc_b = division_title_probs(pre, schedule_played, teams, N_SIMS)
    sc_c = division_title_probs(base, schedule_played, teams, N_SIMS)

    market_raw = {t: american_to_prob(o) for t, o in MARKET_AMERICAN.items()}
    market_norm = None
    if set(market_raw) == set(teams):
        total = sum(market_raw.values())
        market_norm = {t: p / total for t, p in market_raw.items()}

    out = []
    for t in teams:
        out.append({
            "team": t,
            "A_preseason": round(sc_a[t], 4),
            "B_actual": round(sc_b[t], 4),
            "C_actual_sin_luck": round(sc_c[t], 4),
            "mercado_crudo": round(market_raw[t], 4) if t in market_raw else None,
            "mercado_sin_vig": round(market_norm[t], 4) if market_norm else None,
        })
    print("\n=== Probabilidad de titulo de division ===")
    print(pl.DataFrame(out))