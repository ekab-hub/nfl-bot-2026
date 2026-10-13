"""
Paso 3: afinar el K de Elo dentro de la temporada y revisar calibracion.
Para cada temporada 2005-2025, checkpoints semana 4 y 8, y cada K, simula el
resto de la temporada con resultados reales hasta el checkpoint y mide
log-loss / Brier del campeon de division. Tambien arma una tabla de
calibracion: cuando el modelo da X%, que tan seguido ocurre.
"""
import os
import pickle
import time
import numpy as np
import polars as pl
from season_sim_core import (get_season_schedule, simulate_one_season, get_all_division_winners,
                              DIVISIONS, TEAM_TO_DIVISION)
from validar_condicionada import with_retry, base_ratings, mask_schedule, paired

TEST_SEASONS = range(2005, 2026)
CHECKPOINTS = [4, 8]
K_VALUES = [20, 35, 50]
N_SIMS = 3000
CACHE_FILE = "k_calibracion_cache.pkl"
META = (tuple(CHECKPOINTS), tuple(K_VALUES), N_SIMS)
BINS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 1.01]


def division_probs(ratings, schedule, n_sims, seed, k):
    np.random.seed(seed)
    wins = {t: 0 for t in TEAM_TO_DIVISION}
    for _ in range(n_sims):
        records, _ = simulate_one_season(ratings, schedule, k=k)
        for w in get_all_division_winners(records).values():
            wins[w] += 1
    return {t: v / n_sims for t, v in wins.items()}


def run_season(season):
    def prepare():
        return base_ratings(season), get_season_schedule(season, include_played_results=True)

    base, schedule = with_retry(prepare)
    records_real, _ = simulate_one_season(base, schedule)
    winners = get_all_division_winners(records_real)

    out = {}
    for cp in CHECKPOINTS:
        masked = mask_schedule(schedule, cp)
        for k in K_VALUES:
            probs = division_probs(base, masked, N_SIMS, seed=season, k=k)
            ll, br, pairs = [], [], []
            for div, teams in DIVISIONS.items():
                w = winners[div]
                ll.append(-np.log(max(probs[w], 1e-4)))
                br.append(sum((probs[t] - (1.0 if t == w else 0.0)) ** 2 for t in teams))
                pairs.extend((probs[t], 1.0 if t == w else 0.0) for t in teams)
            out[(cp, k)] = {"ll": ll, "br": br, "pairs": pairs}
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

    def collect(cp, k, key):
        return np.array([x for s in seasons_done for x in cache["seasons"][s][(cp, k)][key]])

    rows = []
    for cp in CHECKPOINTS:
        ref = collect(cp, 20, "ll")
        for k in K_VALUES:
            ll, br = collect(cp, k, "ll"), collect(cp, k, "br")
            d, se = paired(ll, ref) if k != 20 else (0.0, 0.0)
            rows.append({
                "semana": cp, "K": k,
                "logloss": round(ll.mean(), 4), "brier": round(br.mean(), 4),
                "dif_vs_K20": round(d, 4), "se": round(se, 4),
            })
    print(f"\n{len(seasons_done)} temporadas. dif negativa = mejor que K=20; significativa si |dif| > 2*se\n")
    with pl.Config(tbl_rows=10, tbl_cols=10):
        print(pl.DataFrame(rows))

    for cp in CHECKPOINTS:
        best_k = min(K_VALUES, key=lambda k: collect(cp, k, "ll").mean())
        pairs = np.array([p for s in seasons_done for p in cache["seasons"][s][(cp, best_k)]["pairs"]])
        p, y = pairs[:, 0], pairs[:, 1]
        rel = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() > 0:
                rel.append({
                    "rango_prob": f"{lo:.0%}-{min(hi, 1.0):.0%}",
                    "n": int(m.sum()),
                    "prob_modelo": round(float(p[m].mean()), 3),
                    "frecuencia_real": round(float(y[m].mean()), 3),
                })
        print(f"\nCalibracion semana {cp} (K={best_k})")
        with pl.Config(tbl_rows=10, tbl_cols=10):
            print(pl.DataFrame(rel))