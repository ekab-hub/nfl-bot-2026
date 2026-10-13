"""
Probabilidad de campeon de Super Bowl, via Monte Carlo. Simula temporada
regular + playoffs completos. use_played=True incorpora los resultados
reales ya jugados y simula solo lo que falta, con K de Elo en temporada.
"""
import time
import polars as pl
from season_sim_core import (get_preseason_ratings, get_season_schedule, simulate_one_season,
                              get_all_division_winners, simulate_playoffs, TEAM_TO_DIVISION,
                              IN_SEASON_K, PRESEASON_K, apply_elo_adjustments)
from qb_out_2026 import qb_elo_adjustments


def run_superbowl_monte_carlo(target_season, n_sims=10000, use_played=False):
    preseason_ratings = get_preseason_ratings(target_season)
    schedule = get_season_schedule(target_season, include_played_results=use_played)
    k = IN_SEASON_K if use_played else PRESEASON_K
    # QB fuera por la temporada: solo en vivo (use_played=True); backtests sin cambios
    elo_adjustments = qb_elo_adjustments(target_season) if use_played else {}

    if use_played:
        n_played = sum(1 for g in schedule if g.get("home_score") is not None)
        print(f"Partidos ya jugados incorporados: {n_played} (K en temporada = {k})")

    sb_wins = {team: 0 for team in TEAM_TO_DIVISION}

    for _ in range(n_sims):
        records, final_elo = simulate_one_season(preseason_ratings, schedule, k=k, elo_adjustments=elo_adjustments)
        division_winners = get_all_division_winners(records)
        sb_champion = simulate_playoffs(records, division_winners, apply_elo_adjustments(final_elo, elo_adjustments))
        sb_wins[sb_champion] += 1

    results = pl.DataFrame([
        {"team": team, "division": TEAM_TO_DIVISION[team], "sb_champion_prob": wins / n_sims}
        for team, wins in sb_wins.items()
    ]).sort("sb_champion_prob", descending=True)

    return results


if __name__ == "__main__":
    TARGET_SEASON = 2026

    t0 = time.time()
    results = run_superbowl_monte_carlo(TARGET_SEASON, n_sims=10000, use_played=True)
    elapsed = time.time() - t0
    print(f"Tiempo total: {elapsed:.2f} segundos\n")

    with pl.Config(tbl_rows=32):
        print(results)

    print(f"\nSuma de sb_champion_prob (deberia ser ~1.0): {results['sb_champion_prob'].sum():.4f}")