"""
Backtest: que tan bien el bot identifica, ANTES de que empiece la temporada,
a los equipos que de verdad van a llegar lejos en playoffs -- extendido al
maximo de historico disponible.

Definicion de "top real" (funciona IGUAL para cualquier año, formato viejo
o nuevo, porque el numero de equipos que llega a cada ronda no cambio nunca):
  - top 8 real = los 8 equipos que llegaron a Ronda Divisional
  - top 4 real = los 4 equipos que llegaron a Conference Championship
  - top 2 real = los 2 equipos que llegaron al Super Bowl
Se saca directo del calendario real (game_type DIV/CON/SB) -- no requiere
ningun dato manual, es 100% confiable para cualquier temporada.

Definicion de "top del bot": 10,000 simulaciones de Monte Carlo con los
preseason ratings (SIN ver ningun resultado de esa temporada), contando en
cuantas simulaciones cada equipo llega a cada ronda. IMPORTANTE: el top N
del bot se selecciona POR CONFERENCIA (N/2 de AFC + N/2 de NFC), no
globalmente entre los 32 equipos -- el bracket real siempre reparte los
lugares 4+4 (Divisional), 2+2 (Conference Champ) y 1+1 (Super Bowl) entre
las dos conferencias, asi que rankear a los 32 juntos podia darle al bot un
"top 4" de puros equipos de una sola conferencia, algo imposible en la
realidad. Se corrige seleccionando el top N/2 dentro de cada conferencia.

DOS FORMATOS DE BRACKET (a diferencia del top real, el ARMADO del bracket
si cambio, y hay que simularlo distinto para que la comparacion sea justa):
  - 2020 en adelante: 7 seeds por conferencia, 1 bye, 3 partidos de wildcard.
  - Antes de 2020 (usamos 2005-2019): 6 seeds por conferencia, 2 byes,
    2 partidos de wildcard.

ALCANCE: 2005-2025 (21 temporadas). Empieza en 2005 porque usamos 2002
como arranque del historial (ultima realineacion de divisiones a la
estructura actual de 32 equipos / 8 divisiones) mas 3 temporadas de
calentamiento del Elo antes de evaluar -- mismo criterio que el resto del
proyecto (backtest_elo_v1.py uso 2015 + 3 años de calentamiento).
"""
import numpy as np
import polars as pl
import nflreadpy as nfl
from season_sim_core import (
    get_preseason_ratings, get_season_schedule, simulate_one_season,
    get_all_division_winners, simulate_playoff_game,
    DIVISIONS, TEAM_TO_DIVISION, CONFERENCES, normalize_team
)

HISTORY_START = 2002
TEST_SEASONS = range(2005, 2026)
N_SIMS = 10000
NEW_FORMAT_FROM = 2020

AFC_TEAMS = {t for t, d in TEAM_TO_DIVISION.items() if d.startswith("AFC")}
NFC_TEAMS = {t for t, d in TEAM_TO_DIVISION.items() if d.startswith("NFC")}


# ---------- Formato NUEVO (2020+): 7 seeds, 1 bye, 3 wildcards ----------
def seed_conference_new(records, conference, division_winners):
    conf_divisions = CONFERENCES[conference]
    conf_teams = [t for d in conf_divisions for t in DIVISIONS[d]]
    winners = [division_winners[d] for d in conf_divisions]

    def win_pct(team):
        r = records[team]
        total = r["w"] + r["l"] + r["t"]
        return (r["w"] + 0.5 * r["t"]) / total if total > 0 else 0

    def point_diff(team):
        r = records[team]
        return r["pf"] - r["pa"]

    winners_sorted = sorted(winners, key=lambda t: (win_pct(t), point_diff(t)), reverse=True)
    non_winners = [t for t in conf_teams if t not in winners]
    wildcards_sorted = sorted(non_winners, key=lambda t: (win_pct(t), point_diff(t)), reverse=True)[:3]
    return winners_sorted + wildcards_sorted  # 7 seeds


def simulate_playoffs_new_tracked(records, division_winners, final_elo, home_advantage=35, margin_std=13.5):
    divisional_entrants, conf_champ_entrants, sb_entrants = [], [], []

    for conf in ["AFC", "NFC"]:
        seeds = seed_conference_new(records, conf, division_winners)
        rank = {team: i + 1 for i, team in enumerate(seeds)}
        s1, s2, s3, s4, s5, s6, s7 = seeds

        w_27 = simulate_playoff_game(s2, s7, final_elo, home_advantage, margin_std)
        w_36 = simulate_playoff_game(s3, s6, final_elo, home_advantage, margin_std)
        w_45 = simulate_playoff_game(s4, s5, final_elo, home_advantage, margin_std)
        divisional_entrants.extend([s1, w_27, w_36, w_45])

        wc_sorted = sorted([w_27, w_36, w_45], key=lambda t: rank[t])
        best_wc, mid_wc, worst_wc = wc_sorted

        div1 = simulate_playoff_game(s1, worst_wc, final_elo, home_advantage, margin_std)
        d2h, d2a = (best_wc, mid_wc) if rank[best_wc] < rank[mid_wc] else (mid_wc, best_wc)
        div2 = simulate_playoff_game(d2h, d2a, final_elo, home_advantage, margin_std)
        conf_champ_entrants.extend([div1, div2])

        cch, cca = (div1, div2) if rank[div1] < rank[div2] else (div2, div1)
        conf_champ = simulate_playoff_game(cch, cca, final_elo, home_advantage, margin_std)
        sb_entrants.append(conf_champ)

    sb_champion = simulate_playoff_game(sb_entrants[0], sb_entrants[1], final_elo,
                                         home_advantage=0, margin_std=margin_std, neutral=True)
    return {"divisional_entrants": divisional_entrants, "conf_champ_entrants": conf_champ_entrants,
            "sb_entrants": sb_entrants, "sb_champion": sb_champion}


# ---------- Formato VIEJO (hasta 2019): 6 seeds, 2 byes, 2 wildcards ----------
def seed_conference_old(records, conference, division_winners):
    conf_divisions = CONFERENCES[conference]
    conf_teams = [t for d in conf_divisions for t in DIVISIONS[d]]
    winners = [division_winners[d] for d in conf_divisions]

    def win_pct(team):
        r = records[team]
        total = r["w"] + r["l"] + r["t"]
        return (r["w"] + 0.5 * r["t"]) / total if total > 0 else 0

    def point_diff(team):
        r = records[team]
        return r["pf"] - r["pa"]

    winners_sorted = sorted(winners, key=lambda t: (win_pct(t), point_diff(t)), reverse=True)
    non_winners = [t for t in conf_teams if t not in winners]
    wildcards_sorted = sorted(non_winners, key=lambda t: (win_pct(t), point_diff(t)), reverse=True)[:2]
    return winners_sorted + wildcards_sorted  # 6 seeds


def simulate_playoffs_old_tracked(records, division_winners, final_elo, home_advantage=35, margin_std=13.5):
    divisional_entrants, conf_champ_entrants, sb_entrants = [], [], []

    for conf in ["AFC", "NFC"]:
        seeds = seed_conference_old(records, conf, division_winners)
        rank = {team: i + 1 for i, team in enumerate(seeds)}
        s1, s2, s3, s4, s5, s6 = seeds

        # Wildcard: (3 vs 6) y (4 vs 5). Seeds 1 y 2 tienen bye.
        w_36 = simulate_playoff_game(s3, s6, final_elo, home_advantage, margin_std)
        w_45 = simulate_playoff_game(s4, s5, final_elo, home_advantage, margin_std)
        divisional_entrants.extend([s1, s2, w_36, w_45])  # 4 por conferencia

        wc_sorted = sorted([w_36, w_45], key=lambda t: rank[t])
        best_wc, worst_wc = wc_sorted

        # Divisional: (1) vs el peor seed sobreviviente, (2) vs el otro
        div1 = simulate_playoff_game(s1, worst_wc, final_elo, home_advantage, margin_std)
        div2 = simulate_playoff_game(s2, best_wc, final_elo, home_advantage, margin_std)
        conf_champ_entrants.extend([div1, div2])

        cch, cca = (div1, div2) if rank[div1] < rank[div2] else (div2, div1)
        conf_champ = simulate_playoff_game(cch, cca, final_elo, home_advantage, margin_std)
        sb_entrants.append(conf_champ)

    sb_champion = simulate_playoff_game(sb_entrants[0], sb_entrants[1], final_elo,
                                         home_advantage=0, margin_std=margin_std, neutral=True)
    return {"divisional_entrants": divisional_entrants, "conf_champ_entrants": conf_champ_entrants,
            "sb_entrants": sb_entrants, "sb_champion": sb_champion}


def top_n_by_conference(counts, n_per_conference):
    """
    Selecciona los N equipos con mas apariciones POR CONFERENCIA, no
    globalmente -- respeta la estructura real del bracket (el Super Bowl
    siempre tiene 1 AFC + 1 NFC, Conference Champ siempre 2+2, Divisional
    siempre 4+4). Rankear a los 32 equipos juntos podia darle al bot un
    "top 4" de puros equipos AFC, algo que en la realidad es imposible.
    """
    afc_counts = {t: c for t, c in counts.items() if t in AFC_TEAMS}
    nfc_counts = {t: c for t, c in counts.items() if t in NFC_TEAMS}
    top_afc = sorted(afc_counts, key=afc_counts.get, reverse=True)[:n_per_conference]
    top_nfc = sorted(nfc_counts, key=nfc_counts.get, reverse=True)[:n_per_conference]
    return set(top_afc) | set(top_nfc)


def get_bot_top_n(target_season, n_sims=N_SIMS):
    preseason_ratings = get_preseason_ratings(target_season, history_start=HISTORY_START)
    schedule = get_season_schedule(target_season)
    playoff_fn = simulate_playoffs_new_tracked if target_season >= NEW_FORMAT_FROM else simulate_playoffs_old_tracked

    divisional_counts = {team: 0 for team in TEAM_TO_DIVISION}
    conf_champ_counts = {team: 0 for team in TEAM_TO_DIVISION}
    sb_counts = {team: 0 for team in TEAM_TO_DIVISION}

    for _ in range(n_sims):
        records, final_elo = simulate_one_season(preseason_ratings, schedule)
        division_winners = get_all_division_winners(records)
        result = playoff_fn(records, division_winners, final_elo)

        for team in result["divisional_entrants"]:
            divisional_counts[team] += 1
        for team in result["conf_champ_entrants"]:
            conf_champ_counts[team] += 1
        for team in result["sb_entrants"]:
            sb_counts[team] += 1

    top8 = top_n_by_conference(divisional_counts, n_per_conference=4)  # 4+4
    top4 = top_n_by_conference(conf_champ_counts, n_per_conference=2)  # 2+2
    top2 = top_n_by_conference(sb_counts, n_per_conference=1)         # 1+1
    return top8, top4, top2


def get_real_top_n(target_season):
    df = nfl.load_schedules(seasons=[target_season])
    df = df.with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])

    def teams_in(game_type):
        g = df.filter(pl.col("game_type") == game_type)
        return set(g["home_team"].to_list()) | set(g["away_team"].to_list())

    return teams_in("DIV"), teams_in("CON"), teams_in("SB")


def hit_rate(real_set, bot_set):
    return len(real_set & bot_set) / len(real_set)


def jaccard(real_set, bot_set):
    return len(real_set & bot_set) / len(real_set | bot_set)


if __name__ == "__main__":
    all_results = []

    for season in TEST_SEASONS:
        fmt = "NUEVO (7 seeds)" if season >= NEW_FORMAT_FROM else "VIEJO (6 seeds)"
        print(f"Simulando temporada {season} [formato {fmt}]...")

        bot_top8, bot_top4, bot_top2 = get_bot_top_n(season)
        real_top8, real_top4, real_top2 = get_real_top_n(season)

        row = {
            "season": season, "formato": fmt,
            "hit_top8": hit_rate(real_top8, bot_top8), "jaccard_top8": jaccard(real_top8, bot_top8),
            "hit_top4": hit_rate(real_top4, bot_top4), "jaccard_top4": jaccard(real_top4, bot_top4),
            "hit_top2": hit_rate(real_top2, bot_top2), "jaccard_top2": jaccard(real_top2, bot_top2),
        }
        all_results.append(row)

        print(f"  Top 8 real: {sorted(real_top8)}")
        print(f"  Top 8 bot : {sorted(bot_top8)}")
        print(f"  Top 4 real: {sorted(real_top4)}  |  Top 4 bot: {sorted(bot_top4)}")
        print(f"  Top 2 real: {sorted(real_top2)}  |  Top 2 bot: {sorted(bot_top2)}")
        print(f"  hit_rate: top8={row['hit_top8']:.2f} top4={row['hit_top4']:.2f} top2={row['hit_top2']:.2f}")
        print(f"  jaccard : top8={row['jaccard_top8']:.2f} top4={row['jaccard_top4']:.2f} top2={row['jaccard_top2']:.2f}\n")

    results_df = pl.DataFrame(all_results)
    results_df.write_parquet("top_n_backtest_results.parquet")

    print("=" * 70)
    print(f"RESUMEN — {len(list(TEST_SEASONS))} temporadas evaluadas ({min(TEST_SEASONS)}-{max(TEST_SEASONS)})")
    print("=" * 70)
    with pl.Config(tbl_rows=30, tbl_cols=10):
        print(results_df)

    print(f"\nPromedio hit_rate  -> top8: {results_df['hit_top8'].mean():.3f}  "
          f"top4: {results_df['hit_top4'].mean():.3f}  top2: {results_df['hit_top2'].mean():.3f}")
    print(f"Promedio jaccard   -> top8: {results_df['jaccard_top8'].mean():.3f}  "
          f"top4: {results_df['jaccard_top4'].mean():.3f}  top2: {results_df['jaccard_top2'].mean():.3f}")

    print("\n--- Promedio por formato (viejo vs nuevo), para ver si hay diferencia ---")
    for fmt_label in results_df["formato"].unique().to_list():
        sub = results_df.filter(pl.col("formato") == fmt_label)
        print(f"{fmt_label} (n={sub.height}): hit_top8={sub['hit_top8'].mean():.3f}  "
              f"hit_top4={sub['hit_top4'].mean():.3f}  hit_top2={sub['hit_top2'].mean():.3f}")