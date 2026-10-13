"""
Parte 1: puntos de spread que el mercado asigna a la disponibilidad del QB, mas alla del Elo.
Ver PROTOCOLO_QB.md. Escribe data/qb_mercado.json y data/qb_mercado_juegos.parquet.
"""
import json
import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np
import polars as pl
import nflreadpy as nfl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from qb_adjustment import qb_career_value   # produccion, solo lectura
from motor_elo_v1 import run_elo_over_seasons, normalize_team
import features as F

DATA = HERE / "data"


def norm_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


# ----------------------------------------------------------------- datos
def qb_values():
    cache = DATA / "qb_career_value_prod.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    pbp = pl.read_parquet(DATA / "pbp.parquet")
    v = qb_career_value(pbp).select(["season", "qb_id", "qb_value_epa_dropback"])
    # produccion agrupa por (id, nombre): si hay dos grafias en una temporada hay dos filas;
    # se promedian (misma temporada, mismo QB)
    v = v.group_by(["season", "qb_id"]).agg(pl.col("qb_value_epa_dropback").mean())
    v.write_parquet(cache)
    return v


def value_lookup(vals, picks):
    """f(qb_id, season) -> valor con datos < season; prior de draft sin fuga."""
    qp = picks.filter(pl.col("position") == "QB").select(
        pl.col("season").cast(pl.Int32).alias("ds"), "round", pl.col("gsis_id").alias("qb_id")).drop_nulls("qb_id").unique("qb_id")
    bucket = {r["qb_id"]: ("R1" if r["round"] == 1 else "R2_R3" if r["round"] in (2, 3) else "R4_PLUS")
              for r in qp.iter_rows(named=True)}
    rookie = vals.join(qp, left_on=["season", "qb_id"], right_on=["ds", "qb_id"])
    rookie = rookie.with_columns(pl.col("qb_id").replace_strict(bucket, default="R4_PLUS").alias("b"))
    hist = {}
    for s, q, v in vals.sort("season").select(["season", "qb_id", "qb_value_epa_dropback"]).iter_rows():
        hist.setdefault(q, []).append((s, v))
    prior_cache = {}

    def prior(season, b):
        if (season, b) not in prior_cache:
            r = rookie.filter((pl.col("season") < season) & (pl.col("b") == b))["qb_value_epa_dropback"]
            prior_cache[(season, b)] = float(r.mean()) if len(r) else float(rookie["qb_value_epa_dropback"].mean())
        return prior_cache[(season, b)]

    def f(qb, season):
        prev = [v for s, v in hist.get(qb, []) if s < season]
        if prev:
            return prev[-1], False
        return prior(season, bucket.get(qb, "R4_PLUS")), True
    return f


def build_games():
    sch = nfl.load_schedules(seasons=list(range(2009, 2026))).filter(pl.col("game_type") == "REG").with_columns(
        pl.col("home_team").replace(F.TEAM_MAP), pl.col("away_team").replace(F.TEAM_MAP))
    elo, _ = run_elo_over_seasons(list(range(1999, 2026)), k=20, home_advantage=35, regression_to_mean=0.40)
    elo = elo.filter(pl.col("game_type") == "REG").select(["season", "week", "home_team", "away_team",
                                                           "pre_elo_home", "pre_elo_away"])
    picks = nfl.load_draft_picks()
    val = value_lookup(qb_values(), picks)

    # QB habitual: lider de titularidades en s-1 (empate -> el mas reciente)
    long = pl.concat([sch.select(["season", "week", pl.col("home_team").alias("team"), pl.col("home_qb_id").alias("qb")]),
                      sch.select(["season", "week", pl.col("away_team").alias("team"), pl.col("away_qb_id").alias("qb")])])
    lead = long.drop_nulls("qb").group_by(["season", "team", "qb"]).agg(pl.len().alias("n"), pl.col("week").max().alias("last")) \
               .sort(["n", "last"], descending=True).unique(["season", "team"], keep="first")
    habitual = {(s + 1, t): q for s, t, q in lead.select(["season", "team", "qb"]).iter_rows()}
    wk1 = {(s, t): q for s, t, q in long.sort("week").unique(["season", "team"], keep="first")
           .select(["season", "team", "qb"]).iter_rows()}

    g = sch.filter(pl.col("season") >= 2010).join(elo, on=["season", "week", "home_team", "away_team"], how="left")
    assert g["pre_elo_home"].null_count() == 0
    rows = []
    for r in g.sort(["season", "week"]).iter_rows(named=True):
        s = r["season"]; neutral = r["location"] == "Neutral"
        d = {"season": s, "week": r["week"], "game_id": r["game_id"], "home_team": r["home_team"],
             "away_team": r["away_team"], "spread_line": r["spread_line"], "neutral": neutral,
             "elo_margin": (r["pre_elo_home"] + (0 if neutral else 35) - r["pre_elo_away"]) / 25.0,
             "result": r["result"]}
        for side in ("home", "away"):
            t = r[f"{side}_team"]; qb = r[f"{side}_qb_id"]
            hab = habitual.get((s, t)) or wk1.get((s, t))
            v_hab, _ = val(hab, s); v_st, rookie = val(qb, s)
            d[f"{side}_hab"] = hab; d[f"{side}_qb"] = qb
            d[f"{side}_v_hab"] = v_hab; d[f"{side}_v_st"] = v_st
            d[f"{side}_caida"] = 0.0 if qb == hab else v_hab - v_st
            d[f"{side}_cambio"] = qb != hab
        rows.append(d)
    df = pl.DataFrame(rows).with_columns((pl.col("spread_line") - pl.col("elo_margin")).alias("y"))

    # rachas por equipo-temporada: posicion dentro de la racha y semana de inicio
    pos = {}
    for side in ("home", "away"):
        pass
    team_long = pl.concat([
        df.select(["season", "week", "game_id", pl.col("home_team").alias("team"), pl.col("home_cambio").alias("c")]),
        df.select(["season", "week", "game_id", pl.col("away_team").alias("team"), pl.col("away_cambio").alias("c")])
    ]).sort(["season", "team", "week"])
    for (s, t), grp in team_long.group_by(["season", "team"], maintain_order=True):
        k, start = 0, None
        for wk, gid, c in grp.select(["week", "game_id", "c"]).iter_rows():
            if c:
                if k == 0:
                    start = wk
                k += 1
                pos[(gid, t)] = (k, start)
            else:
                k = 0
                pos[(gid, t)] = (0, None)
    df = df.with_columns([
        pl.struct(["game_id", "home_team"]).map_elements(lambda x: pos[(x["game_id"], x["home_team"])][0], return_dtype=pl.Int64).alias("home_k"),
        pl.struct(["game_id", "away_team"]).map_elements(lambda x: pos[(x["game_id"], x["away_team"])][0], return_dtype=pl.Int64).alias("away_k"),
        pl.struct(["game_id", "home_team"]).map_elements(lambda x: pos[(x["game_id"], x["home_team"])][1] or 0, return_dtype=pl.Int64).alias("home_start"),
        pl.struct(["game_id", "away_team"]).map_elements(lambda x: pos[(x["game_id"], x["away_team"])][1] or 0, return_dtype=pl.Int64).alias("away_start"),
    ])
    df = df.with_columns((pl.col("away_caida") - pl.col("home_caida")).alias("x"))
    df.write_parquet(DATA / "qb_mercado_juegos.parquet")
    return df


# ----------------------------------------------------------------- OLS cluster
def ols_cluster(y, X, clusters):
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    cl = np.asarray(clusters)
    G = len(np.unique(cl))
    for c in np.unique(cl):
        m = cl == c
        sc = X[m].T @ e[m]
        meat += np.outer(sc, sc)
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k)
    V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, se, G


def cluster_ids(d):
    use_home = d["home_caida"].to_numpy() != 0
    return np.where(use_home, d["season"].cast(pl.Utf8) + "_" + d["home_team"], d["season"].cast(pl.Utf8) + "_" + d["away_team"])


def samples(df):
    any_change = df.filter(pl.col("home_cambio") | pl.col("away_cambio"))
    f2 = lambda side: (pl.col(f"{side}_k").is_between(1, 2))
    clean = lambda side: (~pl.col(f"{side}_cambio")) | f2(side)
    s2 = df.filter((f2("home") | f2("away")) & clean("home") & clean("away"))
    f2b = lambda side: f2(side) & (pl.col(f"{side}_start") >= 2)
    clean_b = lambda side: (~pl.col(f"{side}_cambio")) | f2b(side)
    s2b = df.filter((f2b("home") | f2b("away")) & clean_b("home") & clean_b("away"))
    return {"i": any_change, "ii": s2, "ii_b": s2b}


def fit_spec(d, name):
    d = d.filter(pl.col("spread_line").is_not_null())
    beta, se, G = ols_cluster(d["y"].to_numpy(), d["x"].to_numpy(), cluster_ids(d))
    z = beta[1] / se[1]
    return {"spec": name, "n": d.height, "clusters": G, "a": beta[0], "b": beta[1], "se_b": se[1],
            "p_unilateral": 1 - norm_cdf(z), "x_sd": float(d["x"].std())}


def terciles(d, cuts):
    d = d.filter(pl.col("spread_line").is_not_null())
    Xs = []
    for lo, hi in cuts:
        th = (pl.col("home_v_hab") >= lo) & (pl.col("home_v_hab") < hi)
        ta = (pl.col("away_v_hab") >= lo) & (pl.col("away_v_hab") < hi)
        Xs.append(d.select(pl.when(ta).then(pl.col("away_caida")).otherwise(0.0)
                           - pl.when(th).then(pl.col("home_caida")).otherwise(0.0)).to_series().to_numpy())
    beta, se, G = ols_cluster(d["y"].to_numpy(), np.column_stack(Xs), cluster_ids(d))
    return [{"tercil": i + 1, "rango_valor_habitual": [cuts[i][0], cuts[i][1]], "b": beta[i + 1], "se_b": se[i + 1]}
            for i in range(len(cuts))]


def main():
    df = build_games()
    sel = df.filter(pl.col("season") <= 2022); ver = df.filter(pl.col("season") >= 2023)
    out = {"n_partidos": df.height, "seleccion": [], "verificacion": []}
    for part, dd in [("seleccion", sel), ("verificacion", ver)]:
        for name, s in samples(dd).items():
            out[part].append(fit_spec(s, name))
    # terciles sobre la muestra (ii) de seleccion
    s2 = samples(sel)["ii"]
    hv = pl.concat([s2.filter(pl.col("home_caida") != 0)["home_v_hab"], s2.filter(pl.col("away_caida") != 0)["away_v_hab"]])
    q1, q2 = float(hv.quantile(1 / 3)), float(hv.quantile(2 / 3))
    cuts = [(-9.0, q1), (q1, q2), (q2, 9.0)]
    out["terciles_ii"] = terciles(s2, cuts)
    out["terciles_i"] = terciles(samples(sel)["i"], cuts)
    # elite vs suplente
    all_hab = pl.concat([sel["home_v_hab"], sel["away_v_hab"]])
    repl = pl.concat([s2.filter(pl.col("home_caida") != 0)["home_v_st"], s2.filter(pl.col("away_caida") != 0)["away_v_st"]])
    elite, backup = float(all_hab.quantile(0.9)), float(repl.median())
    b2 = [r for r in out["seleccion"] if r["spec"] == "ii"][0]
    out["elite_vs_suplente"] = {"valor_elite_p90": elite, "valor_suplente_mediana": backup, "dv": elite - backup,
                                "puntos": b2["b"] * (elite - backup), "se_puntos": b2["se_b"] * (elite - backup),
                                "elo": 25 * b2["b"] * (elite - backup)}
    b2v = [r for r in out["verificacion"] if r["spec"] == "ii"][0]
    out["puerta"] = {"b_sel": b2["b"], "p_sel": b2["p_unilateral"], "b_ver": b2v["b"],
                     "pasa": bool(b2["b"] > 0 and b2["p_unilateral"] < 0.05 and b2v["b"] > 0)}
    # muestras: sanity
    out["sanity"] = {"frac_partidos_con_cambio": float((df["home_cambio"] | df["away_cambio"]).mean()),
                     "corr_spread_elo": float(np.corrcoef(df.drop_nulls("spread_line")["spread_line"],
                                                          df.drop_nulls("spread_line")["elo_margin"])[0, 1])}
    (DATA / "qb_mercado.json").write_text(json.dumps(out, indent=2, default=float))
    for part in ("seleccion", "verificacion"):
        print(part)
        for r in out[part]:
            print(f"  ({r['spec']:5s}) n={r['n']:5d} clusters={r['clusters']:4d}  b={r['b']:+.3f}  SE={r['se_b']:.3f}  "
                  f"p1={r['p_unilateral']:.2g}  a={r['a']:+.3f}")
    for k in ("terciles_ii", "terciles_i"):
        print(k, [(t["tercil"], [round(c, 3) for c in t["rango_valor_habitual"]], round(t["b"], 2), round(t["se_b"], 2)) for t in out[k]])
    print("elite vs suplente:", out["elite_vs_suplente"])
    print("puerta:", out["puerta"]); print("sanity:", out["sanity"])


if __name__ == "__main__":
    main()
