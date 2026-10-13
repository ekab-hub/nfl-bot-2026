"""
Probabilidad de campeon de conferencia (AFC/NFC): el equipo que gana el
Conference Championship y llega al Super Bowl, sin importar si despues
gana o pierde ahi. use_played=True incorpora los resultados reales ya
jugados y simula solo lo que falta, con K de Elo en temporada.
"""
import time
import polars as pl
from season_sim_core import (get_preseason_ratings, get_season_schedule, simulate_one_season,
                              get_all_division_winners, seed_conference, simulate_playoff_game,
                              TEAM_TO_DIVISION, IN_SEASON_K, PRESEASON_K, apply_elo_adjustments)
from qb_out_2026 import qb_elo_adjustments


def simulate_conference_champion(records, division_winners, final_elo, home_advantage=35, margin_std=13.5):
    conf_champs = {}
    for conf in ["AFC", "NFC"]:
        seeds = seed_conference(records, conf, division_winners)
        rank = {team: i + 1 for i, team in enumerate(seeds)}
        s1, s2, s3, s4, s5, s6, s7 = seeds

        w_27 = simulate_playoff_game(s2, s7, final_elo, home_advantage, margin_std)
        w_36 = simulate_playoff_game(s3, s6, final_elo, home_advantage, margin_std)
        w_45 = simulate_playoff_game(s4, s5, final_elo, home_advantage, margin_std)

        best_wc, mid_wc, worst_wc = sorted([w_27, w_36, w_45], key=lambda t: rank[t])

        div1 = simulate_playoff_game(s1, worst_wc, final_elo, home_advantage, margin_std)
        d2h, d2a = (best_wc, mid_wc) if rank[best_wc] < rank[mid_wc] else (mid_wc, best_wc)
        div2 = simulate_playoff_game(d2h, d2a, final_elo, home_advantage, margin_std)

        cch, cca = (div1, div2) if rank[div1] < rank[div2] else (div2, div1)
        conf_champs[conf] = simulate_playoff_game(cch, cca, final_elo, home_advantage, margin_std)
    return conf_champs


def run_conference_monte_carlo(target_season, n_sims=10000, use_played=False):
    preseason_ratings = get_preseason_ratings(target_season)
    schedule = get_season_schedule(target_season, include_played_results=use_played)
    k = IN_SEASON_K if use_played else PRESEASON_K
    # QB fuera por la temporada: solo en vivo (use_played=True); backtests sin cambios
    elo_adjustments = qb_elo_adjustments(target_season) if use_played else {}

    if use_played:
        n_played = sum(1 for g in schedule if g.get("home_score") is not None)
        print(f"Partidos ya jugados incorporados: {n_played} (K en temporada = {k})")

    conf_champ_wins = {team: 0 for team in TEAM_TO_DIVISION}

    for _ in range(n_sims):
        records, final_elo = simulate_one_season(preseason_ratings, schedule, k=k, elo_adjustments=elo_adjustments)
        division_winners = get_all_division_winners(records)
        conf_champs = simulate_conference_champion(records, division_winners,
                                                   apply_elo_adjustments(final_elo, elo_adjustments))
        for winner in conf_champs.values():
            conf_champ_wins[winner] += 1

    results = pl.DataFrame([
        {"team": team, "division": TEAM_TO_DIVISION[team], "conference_champion_prob": wins / n_sims}
        for team, wins in conf_champ_wins.items()
    ]).sort("conference_champion_prob", descending=True)

    return results


if __name__ == "__main__":
    TARGET_SEASON = 2026

    t0 = time.time()
    results = run_conference_monte_carlo(TARGET_SEASON, n_sims=10000, use_played=True)
    elapsed = time.time() - t0
    print(f"Tiempo total: {elapsed:.2f} segundos\n")

    afc_results = results.filter(pl.col("division").str.starts_with("AFC")).sort("conference_champion_prob", descending=True)
    nfc_results = results.filter(pl.col("division").str.starts_with("NFC")).sort("conference_champion_prob", descending=True)

    print("=== AFC — probabilidad de campeón de conferencia ===")
    with pl.Config(tbl_rows=16):
        print(afc_results)

    print("\n=== NFC — probabilidad de campeón de conferencia ===")
    with pl.Config(tbl_rows=16):
        print(nfc_results)

    print(f"\nSuma AFC (debería ser ~1.0): {afc_results['conference_champion_prob'].sum():.4f}")
    print(f"Suma NFC (debería ser ~1.0): {nfc_results['conference_champion_prob'].sum():.4f}")