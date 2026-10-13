"""
Paso 1: calibrar la escala del ajuste de luck (WINS_TO_ELO_SCALE) contra
resultados historicos. Para cada temporada 2005-2025 y cada escala, simula
la temporada desde preseason (sin ver resultados) y mide que tan bien la
probabilidad de campeon de division predijo al campeon real.
Metricas: log-loss y Brier multiclase por division-temporada (8 x 21 = 168),
con diferencias pareadas contra escala 0 (sin luck).
Reintenta descargas fallidas y guarda progreso por temporada en un cache.
"""
import os
import pickle
import time
import numpy as np
import polars as pl
from motor_elo_v1 import run_elo_over_seasons
from underlying_stats import build_team_season_stats
from season_sim_core import (get_season_schedule, simulate_one_season, get_all_division_winners,
                              DIVISIONS, TEAM_TO_DIVISION, LUCK_COEF)

HISTORY_START = 2002
TEST_SEASONS = range(2005, 2026)
SCALES = [0, 20, 30, 40, 50, 65]
N_SIMS = 3000
EXTREME_SHIFT = 60
CACHE_FILE = "calibrar_luck_cache.pkl"
META = (tuple(SCALES), N_SIMS, EXTREME_SHIFT, HISTORY_START)


def with_retry(fn, tries=5, wait=15):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  fallo de descarga ({type(e).__name__}), reintento {i + 1}/{tries - 1} en {wait * (i + 1)}s...")
            time.sleep(wait * (i + 1))


def season_inputs(season):
    _, engine = run_elo_over_seasons(list(range(HISTORY_START, season)), home_advantage=35)
    engine.new_season_regression()
    base = dict(engine.ratings)
    luck = {r["team"]: r["luck"] for r in build_team_season_stats([season - 1]).iter_rows(named=True)}
    return base, luck


def ratings_for(base, luck, scale):
    out = dict(base)
    for team, lk in luck.items():
        out[team] = out.get(team, 1500) + LUCK_COEF * lk * scale
    return out


def actual_division_winners(season, base):
    schedule = get_season_schedule(season, include_played_results=True)
    records, _ = simulate_one_season(base, schedule)
    return get_all_division_winners(records)


def division_probs(ratings, schedule, n_sims, seed):
    np.random.seed(seed)
    wins = {t: 0 for t in TEAM_TO_DIVISION}
    for _ in range(n_sims):
        records, _ = simulate_one_season(ratings, schedule)
        for winner in get_all_division_winners(records).values():
            wins[winner] += 1
    return {t: w / n_sims for t, w in wins.items()}


def prepare_season(season):
    base, luck = season_inputs(season)
    schedule = get_season_schedule(season)
    winners = actual_division_winners(season, base)
    return base, luck, schedule, winners


def run_season(season):
    base, luck, schedule, winners = with_retry(lambda: prepare_season(season))
    shifts65 = {t: LUCK_COEF * lk * 65 for t, lk in luck.items()}

    out = {"extreme": [], "ll": {s: [] for s in SCALES}, "br": {s: [] for s in SCALES}}
    for div, teams in DIVISIONS.items():
        out["extreme"].append(max(abs(shifts65.get(t, 0.0)) for t in teams) > EXTREME_SHIFT)

    for s in SCALES:
        probs = division_probs(ratings_for(base, luck, s), schedule, N_SIMS, seed=season)
        for div, teams in DIVISIONS.items():
            w = winners[div]
            out["ll"][s].append(-np.log(max(probs[w], 1e-4)))
            out["br"][s].append(sum((probs[t] - (1.0 if t == w else 0.0)) ** 2 for t in teams))
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
    extreme = np.array([x for s in seasons_done for x in cache["seasons"][s]["extreme"]])
    ll = {sc: np.array([x for s in seasons_done for x in cache["seasons"][s]["ll"][sc]]) for sc in SCALES}
    br = {sc: np.array([x for s in seasons_done for x in cache["seasons"][s]["br"][sc]]) for sc in SCALES}

    rows = []
    for sc in SCALES:
        d_all, se_all = paired(ll[sc], ll[0]) if sc != 0 else (0.0, 0.0)
        d_ext, se_ext = paired(ll[sc][extreme], ll[0][extreme]) if sc != 0 else (0.0, 0.0)
        rows.append({
            "escala": sc,
            "logloss": round(ll[sc].mean(), 4),
            "brier": round(br[sc].mean(), 4),
            "dif_vs_0": round(d_all, 4),
            "se": round(se_all, 4),
            "dif_extremas": round(d_ext, 4),
            "se_extremas": round(se_ext, 4),
        })

    print(f"\n{len(seasons_done)} temporadas, {len(extreme)} divisiones-temporada; "
          f"{int(extreme.sum())} con algun equipo con |ajuste luck a escala 65| > {EXTREME_SHIFT}")
    print("dif negativa = mejor que NO usar luck. Es significativa si |dif| > 2*se.\n")
    with pl.Config(tbl_rows=20, tbl_cols=10):
        print(pl.DataFrame(rows))