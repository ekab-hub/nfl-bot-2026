"""
Paso 2: validar la proyeccion que incorpora resultados ya jugados.
Para cada temporada 2005-2025 y cada checkpoint (semana 0, 4, 8) simula el
resto de la temporada condicionando en los resultados reales hasta ese
checkpoint, y mide (a) que tan bien predice al campeon de division
(log-loss y Brier) y (b) hit rate de top 8 / 4 / 2 de playoffs.
Semana 0 = proyeccion de preseason. Sin ajuste de luck.
"""
import os
import pickle
import time
import numpy as np
import polars as pl
from motor_elo_v1 import run_elo_over_seasons
from season_sim_core import (get_season_schedule, simulate_one_season, get_all_division_winners,
                              DIVISIONS, TEAM_TO_DIVISION)
from top_n_backtest import (simulate_playoffs_new_tracked, simulate_playoffs_old_tracked,
                            top_n_by_conference, get_real_top_n, hit_rate, NEW_FORMAT_FROM)

HISTORY_START = 2002
TEST_SEASONS = range(2005, 2026)
CHECKPOINTS = [0, 4, 8]
N_SIMS = 3000
CACHE_FILE = "validar_condicionada_cache.pkl"
META = (tuple(CHECKPOINTS), N_SIMS, HISTORY_START)


def with_retry(fn, tries=5, wait=15):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  fallo de descarga ({type(e).__name__}), reintento {i + 1}/{tries - 1} en {wait * (i + 1)}s...")
            time.sleep(wait * (i + 1))


def base_ratings(season):
    _, engine = run_elo_over_seasons(list(range(HISTORY_START, season)), home_advantage=35)
    engine.new_season_regression()
    return dict(engine.ratings)


def mask_schedule(schedule, checkpoint):
    out = []
    for g in schedule:
        if g["week"] <= checkpoint:
            out.append(g)
        else:
            out.append({"week": g["week"], "home_team": g["home_team"], "away_team": g["away_team"]})
    return out


def run_checkpoint(ratings, schedule, playoff_fn, n_sims, seed):
    np.random.seed(seed)
    div_wins = {t: 0 for t in TEAM_TO_DIVISION}
    divisional = {t: 0 for t in TEAM_TO_DIVISION}
    conf_champ = {t: 0 for t in TEAM_TO_DIVISION}
    sb = {t: 0 for t in TEAM_TO_DIVISION}

    for _ in range(n_sims):
        records, final_elo = simulate_one_season(ratings, schedule)
        division_winners = get_all_division_winners(records)
        for w in division_winners.values():
            div_wins[w] += 1
        result = playoff_fn(records, division_winners, final_elo)
        for t in result["divisional_entrants"]:
            divisional[t] += 1
        for t in result["conf_champ_entrants"]:
            conf_champ[t] += 1
        for t in result["sb_entrants"]:
            sb[t] += 1

    probs = {t: w / n_sims for t, w in div_wins.items()}
    top8 = top_n_by_conference(divisional, 4)
    top4 = top_n_by_conference(conf_champ, 2)
    top2 = top_n_by_conference(sb, 1)
    return probs, top8, top4, top2


def run_season(season):
    def prepare():
        base = base_ratings(season)
        schedule = get_season_schedule(season, include_played_results=True)
        real8, real4, real2 = get_real_top_n(season)
        return base, schedule, real8, real4, real2

    base, schedule, real8, real4, real2 = with_retry(prepare)
    records_real, _ = simulate_one_season(base, schedule)
    winners = get_all_division_winners(records_real)
    playoff_fn = simulate_playoffs_new_tracked if season >= NEW_FORMAT_FROM else simulate_playoffs_old_tracked

    out = {}
    for cp in CHECKPOINTS:
        probs, top8, top4, top2 = run_checkpoint(base, mask_schedule(schedule, cp), playoff_fn, N_SIMS, seed=season)
        ll, br = [], []
        for div, teams in DIVISIONS.items():
            w = winners[div]
            ll.append(-np.log(max(probs[w], 1e-4)))
            br.append(sum((probs[t] - (1.0 if t == w else 0.0)) ** 2 for t in teams))
        out[cp] = {"ll": ll, "br": br,
                   "h8": hit_rate(real8, top8), "h4": hit_rate(real4, top4), "h2": hit_rate(real2, top2)}
    return out


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if cache.get("meta") == META:
            return cache
    return {"meta": META, "seasons": {}}


def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def paired(a, b):
    d = a - b
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


if __name__ == "__main__":
    t0 = time.time()
    cache = load_cache()

    for season in TEST_SEASONS:
        if season in cache["seasons"]:
            print(f"{season} ya estaba en cache")
            continue
        cache["seasons"][season] = run_season(season)
        save_cache(cache)
        print(f"{season} listo ({time.time() - t0:.0f}s)")

    seasons_done = [s for s in TEST_SEASONS if s in cache["seasons"]]

    def collect(cp, key):
        return np.array([x for s in seasons_done for x in cache["seasons"][s][cp][key]])

    base_ll = collect(0, "ll")
    rows = []
    for cp in CHECKPOINTS:
        ll, br = collect(cp, "ll"), collect(cp, "br")
        d, se = paired(ll, base_ll) if cp != 0 else (0.0, 0.0)
        rows.append({
            "semana": cp,
            "logloss": round(ll.mean(), 4),
            "brier": round(br.mean(), 4),
            "dif_vs_preseason": round(d, 4),
            "se": round(se, 4),
            "hit_top8": round(float(np.mean([cache["seasons"][s][cp]["h8"] for s in seasons_done])), 3),
            "hit_top4": round(float(np.mean([cache["seasons"][s][cp]["h4"] for s in seasons_done])), 3),
            "hit_top2": round(float(np.mean([cache["seasons"][s][cp]["h2"] for s in seasons_done])), 3),
        })

    print(f"\n{len(seasons_done)} temporadas, {len(base_ll)} divisiones-temporada")
    print("dif negativa = mejor que la proyeccion de preseason. Azar: top8 0.25, top4 0.125, top2 0.0625\n")
    with pl.Config(tbl_rows=10, tbl_cols=10):
        print(pl.DataFrame(rows))