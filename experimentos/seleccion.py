"""
Etapa de SELECCION (2010-2022). Ver PROTOCOLO.md.
- S0 (spec), cada candidato de uno en uno, Holm sobre la familia (m=18),
- blend con w re-optimizado por version,
- referencias de mercado (V1, V2), conjunto de sobrevivientes.
Escribe resultados.csv (etapa seleccion/referencia/conjunto) y finalistas.json.
No toca 2023-2025.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl
import evalcore as E

HERE = Path(__file__).parent
SEL = list(range(2010, 2023))
df_all = pl.read_parquet(E.DATA / "features.parquet")
DF = df_all.filter(pl.col("season") <= 2022)          # el holdout NO entra aqui
assert DF["season"].max() == 2022

CANDS = {
    "C1a": dict(desc="EPA de pase del ataque propio", add=["team_pass_epa"]),
    "C1b": dict(desc="success rate del ataque propio", add=["team_success"]),
    "C1c": dict(desc="defensa rival separada pase/carrera", add=["opp_def_pass_epa", "opp_def_rush_epa"],
                drop=["opp_def_epa_allowed"]),
    "C2a": dict(desc="entregas con regresion a la media", add=["team_giveaway", "opp_takeaway"]),
    "C2b": dict(desc="sack rate propio y del rival", add=["team_sack_allowed", "opp_sack_rate"]),
    "C2c": dict(desc="presion (participation, 2017+)", add=["team_pressure_allowed", "opp_pressure_rate"], start=2017),
    "C2d": dict(desc="jugadas explosivas", add=["team_explosive", "opp_explosive_allowed"]),
    "C2e": dict(desc="EPA de equipos especiales", add=["team_st_epa", "opp_st_epa"]),
    "C2f": dict(desc="ritmo (jugadas por partido)", add=["team_pace", "opp_pace"]),
    "C3a": dict(desc="techo / superficie", add=["indoor", "turf"]),
    "C3b": dict(desc="clima OBSERVADO (cota optimista)", add=["wind_obs", "cold_obs"]),
    "C4a": dict(desc="titulares propios inactivos OL/WR1 (2013+)", add=["ol_out", "wr1_out"], start=2013),
    "C4b": dict(desc="titulares rivales inactivos DL/CB1 (2013+)", add=["opp_dl_out", "opp_cb1_out"], start=2013),
    "C4c": dict(desc="QB con EPA de la temporada (decay)", add=["qb_value_live"], drop=["team_qb_value"]),
    "C4d": dict(desc="marca de QB suplente", add=["backup_qb"]),
}


def feats_of(c):
    drop = set(c.get("drop", []))
    return [f for f in E.BASE if f not in drop] + c["add"]


def test_seasons(start, seasons=SEL):
    return [s for s in seasons if s >= start + 3]


_cache = {}
def run(feats, start=E.TRAIN_START, seasons=SEL, spec="eval", offset=None, df=None):
    key = (tuple(feats), start, tuple(seasons), spec, offset)
    if key not in _cache:
        _cache[key] = E.walk_forward(df if df is not None else DF, feats, test_seasons(start, seasons),
                                     start, spec, offset)
    return _cache[key]


def game_level(pred, p_fn=None):
    g = E.attach_elo(E.to_games(pred))
    p = E.p_home_skellam(g["lh"].to_numpy(), g["la"].to_numpy()) if p_fn is None else p_fn(g)
    return g, p


def compare(pb, pc, metric_b=None, metric_c=None, pfn_b=None, pfn_c=None, w_b=None, w_c=None):
    """Delta pareado candidato - base. metric_*: vector por equipo-partido (default devianza)."""
    j = pb.join(pc.select(["game_id", "team", pl.col("mu").alias("mu_c")]), on=["game_id", "team"])
    assert j.height == pb.height == pc.height
    y = j["points"].to_numpy().astype(float)
    mb = E.poisson_dev(y, j["mu"].to_numpy()) if metric_b is None else metric_b
    mc = E.poisson_dev(y, j["mu_c"].to_numpy()) if metric_c is None else metric_c
    d, se, z, p = E.paired(mc - mb, E.clusters_of(j))
    gb, p_b = game_level(pb, pfn_b); gc, p_c = game_level(pc, pfn_c)
    assert (gb["game_id"] == gc["game_id"]).all()
    cl_g = E.clusters_of(gb)
    tot = (gb["hp"] + gb["ap"]).to_numpy(); mar = (gb["hp"] - gb["ap"]).to_numpy()
    et_b = (gb["lh"] + gb["la"]).to_numpy() - tot; et_c = (gc["lh"] + gc["la"]).to_numpy() - tot
    em_b = (gb["lh"] - gb["la"]).to_numpy() - mar; em_c = (gc["lh"] - gc["la"]).to_numpy() - mar
    _, _, _, p_tot = E.paired(et_c ** 2 - et_b ** 2, cl_g)
    _, _, _, p_mar = E.paired(em_c ** 2 - em_b ** 2, cl_g)
    bb = E.blend_eval(gb, p_b, w_b); bc = E.blend_eval(gc, p_c, w_c)
    keep = bb["keep"]
    dl, sel_, zl, pll = E.paired((bc["per_game"] - bb["per_game"])[keep], np.array(cl_g)[keep].tolist())
    ll_pois_b = E.logloss_vec(p_b, (gb["hp"] > gb["ap"]).to_numpy().astype(float))[keep].mean()
    ll_pois_c = E.logloss_vec(p_c, (gc["hp"] > gc["ap"]).to_numpy().astype(float))[keep].mean()
    return {
        "n_filas": j.height, "n_partidos": int(keep.sum()),
        "temporadas_test": f"{j['season'].min()}-{j['season'].max()}",
        "dev_base": float(mb.mean()), "dev_cand": float(mc.mean()),
        "delta_dev": d, "se_dev": se, "z": z, "p": p,
        "rmse_total_base": float(np.sqrt((et_b ** 2).mean())), "delta_rmse_total": float(np.sqrt((et_c ** 2).mean()) - np.sqrt((et_b ** 2).mean())), "p_rmse_total": p_tot,
        "rmse_margen_base": float(np.sqrt((em_b ** 2).mean())), "delta_rmse_margen": float(np.sqrt((em_c ** 2).mean()) - np.sqrt((em_b ** 2).mean())), "p_rmse_margen": p_mar,
        "w_base": bb["w"], "w_cand": bc["w"], "w_opt_base": bb["w_opt"], "w_opt_cand": bc["w_opt"], "logloss_base": bb["logloss"], "logloss_cand": bc["logloss"],
        "delta_logloss": dl, "se_logloss": sel_, "p_logloss": pll,
        "acc_base": bb["acc"], "acc_cand": bc["acc"],
        "logloss_poisson_solo_base": float(ll_pois_b), "logloss_poisson_solo_cand": float(ll_pois_c),
    }


def leak_ok(feats):
    lr = pl.read_csv(E.DATA / "leakage_result.csv")
    ok = dict(zip(lr["feature"], lr["pasa"]))
    return all(ok.get(f, True) for f in feats)


def main():
    rows = []

    # ---------------- referencia: reproduccion de produccion (spec prod, w=0.25 fijo)
    pp, _ = run(E.BASE, spec="prod")
    g, p = game_level(pp)
    b25 = E.blend_eval(g, p, 0.25)
    y = (g["hp"] > g["ap"]).to_numpy().astype(float); keep = b25["keep"]
    pe = g["p_elo"].to_numpy()
    rows.append({"etapa": "referencia", "id": "PROD", "descripcion": "baseline spec produccion, blend w=0.25 fijo",
                 "features": ",".join(E.BASE), "temporadas_test": "2010-2022", "n_partidos": int(keep.sum()),
                 "logloss_cand": b25["logloss"], "acc_cand": b25["acc"], "w_cand": 0.25,
                 "logloss_base": float(E.logloss_vec(pe, y)[keep].mean()),
                 "acc_base": float(np.mean(((pe > 0.5) == (y == 1))[keep])),
                 "decision": "referencia (base = Elo solo)"})

    # ---------------- S0 y candidatos
    fam = []
    pe_, _ = run(E.BASE)
    r = compare(pp, pe_); r.update(id="S0", descripcion="features estandarizadas + alpha=1e-4 (vs spec produccion)",
                                   features=",".join(E.BASE), leakage_ok=True); fam.append(r)
    for cid, c in CANDS.items():
        start = c.get("start", E.TRAIN_START)
        pb, _ = run(E.BASE, start); pc, _ = run(feats_of(c), start)
        r = compare(pb, pc); r.update(id=cid, descripcion=c["desc"], features=",".join(feats_of(c)),
                                      train_start=start, leakage_ok=leak_ok(c["add"]))
        fam.append(r); print(cid, f"{r['delta_dev']:+.5f} p={r['p']:.2g} dLL={r['delta_logloss']:+.5f}")

    # ---------------- O1: binomial negativa
    pb, fits = run(E.BASE)
    j = pb.sort(["game_id", "team"])
    nll_p = -E.poisson.logpmf(j["points"].to_numpy(), j["mu"].to_numpy())
    rs = {s: E.fit_nb_r(f["points"].to_numpy(), f["mu"].to_numpy()) for s, f in fits.items()}
    r_vec = np.array([rs[s] for s in j["season"].to_list()])
    nll_nb = E.nb_nll(j["points"].to_numpy(), j["mu"].to_numpy(), r_vec)
    def pfn_nb(g):
        rr = np.array([rs[s] for s in g["season"].to_list()])
        out = np.empty(g.height)
        for s in set(g["season"].to_list()):
            m = (g["season"] == s).to_numpy()
            out[m] = E.nb_p_home(g["lh"].to_numpy()[m], g["la"].to_numpy()[m], rs[s])
        return out
    r = compare(j, j, metric_b=2 * nll_p, metric_c=2 * nll_nb, pfn_c=pfn_nb)
    r.update(id="O1", descripcion=f"binomial negativa NB2 (r por fold, mediana {np.median(list(rs.values())):.2f}); metrica 2*dNLL",
             features=",".join(E.BASE), leakage_ok=True); fam.append(r); print("O1", r["delta_dev"], r["p"])

    # ---------------- O2: Poisson bivariado (correlacion)
    l3s = {}
    for s, f in fits.items():
        gt = E.to_games(f.with_columns(pl.lit(s).alias("season"), pl.lit(0).alias("week"), pl.lit("").alias("team")))
        l3s[s] = E.fit_l3(gt["hp"].to_numpy(), gt["ap"].to_numpy(), gt["lh"].to_numpy(), gt["la"].to_numpy())
    gg = E.to_games(pb)
    l3v = np.array([l3s[s] for s in gg["season"].to_list()])
    jb = E.bivar_nll(gg["hp"].to_numpy(), gg["ap"].to_numpy(), gg["lh"].to_numpy(), gg["la"].to_numpy(), 0.0)
    jc = np.array([E.bivar_nll(np.array([h]), np.array([a]), np.array([x]), np.array([w]), l)[0]
                   for h, a, x, w, l in zip(gg["hp"].to_numpy(), gg["ap"].to_numpy(), gg["lh"].to_numpy(),
                                            gg["la"].to_numpy(), l3v)])
    # metrica por equipo-partido: 2*dNLL conjunto / 2 asignado a cada fila del partido
    dmap = dict(zip(gg["game_id"].to_list(), (jc - jb).tolist()))
    jj = pb.sort(["game_id", "team"])
    dvec = np.array([dmap[gid] for gid in jj["game_id"].to_list()])
    def pfn_bv(g):
        l = np.array([l3s[s] for s in g["season"].to_list()])
        return E.p_home_skellam(g["lh"].to_numpy(), g["la"].to_numpy(), l)
    r = compare(jj, jj, metric_b=np.zeros(jj.height), metric_c=dvec, pfn_c=pfn_bv)
    r.update(id="O2", descripcion=f"Poisson bivariado, lambda3 por fold {sorted(set(l3s.values()))}; metrica 2*dNLL conjunta/2",
             features=",".join(E.BASE), leakage_ok=True); fam.append(r); print("O2", r["delta_dev"], l3s)

    # ---------------- Holm sobre la familia (m = 18)
    assert len(fam) == 18, len(fam)
    adj = E.holm([r["p"] for r in fam])
    survivors = []
    for r, pa in zip(fam, adj):
        r["p_holm"] = float(pa)
        r["pasa_a"] = bool(pa < 0.01 and r["delta_dev"] < 0)
        r["pasa_c"] = bool(r["delta_logloss"] <= -0.001)
        r["etapa"] = "seleccion"
        r["decision"] = ("finalista (pasa a)" if r["pasa_a"] else "descartado: no pasa (a)")
        if r["pasa_a"] and r["id"].startswith("C"):
            survivors.append(r["id"])
    rows += fam

    # ---------------- conjunto de sobrevivientes (factores C*)
    def joint_feats(lst):
        drop = set().union(*[set(CANDS[c].get("drop", [])) for c in lst]) if lst else set()
        return [f for f in E.BASE if f not in drop] + [f for c in lst for f in CANDS[c]["add"]]

    joints = {}
    deployable = [c for c in survivors if leak_ok(CANDS[c]["add"])]
    for jid, lst in [("JOINT", survivors), ("JOINT_PROD", deployable)]:
        if len(lst) < 2 or (jid == "JOINT_PROD" and lst == survivors):
            continue
        feats = joint_feats(lst)
        start = max(CANDS[c].get("start", E.TRAIN_START) for c in lst)
        pb, _ = run(E.BASE, start); pc, _ = run(feats, start)
        r = compare(pb, pc); r.update(etapa="conjunto", id=jid, descripcion="+".join(lst),
                                      features=",".join(feats), train_start=start,
                                      leakage_ok=leak_ok([f for c in lst for f in CANDS[c]["add"]]),
                                      decision="finalista (conjunto)")
        rows.append(r); joints[jid] = dict(feats=feats, start=start, survivors=lst)
        # aporte marginal de cada sobreviviente dentro del conjunto (leave-one-out)
        for c in lst:
            pr_, _ = run(joint_feats([x for x in lst if x != c]), start)
            r2 = compare(pr_, pc); r2.update(etapa="conjunto", id=f"{jid}-sin-{c}",
                                              descripcion=f"aporte de {c} dado el resto de {jid}", features=",".join(feats),
                                              train_start=start, decision="diagnostico")
            rows.append(r2)

    # ---------------- referencias de mercado
    pb, _ = run(E.BASE)
    pv, _ = run(E.BASE + ["log_implied"])
    r = compare(pb, pv); r.update(etapa="referencia", id="V1", descripcion="baseline + log(total implicito de Vegas): techo",
                                  features=",".join(E.BASE + ["log_implied"]), decision="referencia"); rows.append(r)
    pm, _ = run(["log_implied"])
    r = compare(pb, pm); r.update(etapa="referencia", id="V1m", descripcion="solo mercado: Poisson con log(implicito)",
                                  features="log_implied", decision="referencia"); rows.append(r)
    po, _ = run([], offset="log_implied")
    resid_p = []
    for cid, c in [("BASE", dict(add=E.BASE, desc="features baseline"))] + list(CANDS.items()):
        start = c.get("start", E.TRAIN_START)
        p0, _ = run([], start, offset="log_implied")
        p1, _ = run([f for f in c["add"] if f != "is_home"] if cid != "BASE" else E.BASE, start, offset="log_implied")
        r = compare(p0, p1); r.update(etapa="referencia", id=f"V2-{cid}",
                                      descripcion=f"residual vs mercado: offset log(implicito) + {c['desc']}",
                                      features=",".join(c["add"]), train_start=start, decision="referencia")
        rows.append(r); resid_p.append(r["p"])
    adj = E.holm(resid_p)
    for r, pa in zip([x for x in rows if str(x["id"]).startswith("V2-")], adj):
        r["p_holm"] = float(pa)

    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_csv(HERE / "resultados.csv")
    fin = {"survivors": survivors, "joints": joints,
           "otros_finalistas": [r["id"] for r in fam if r["pasa_a"] and not r["id"].startswith("C")],
           "candidates": {c: dict(feats=feats_of(CANDS[c]), start=CANDS[c].get("start", E.TRAIN_START))
                          for c in survivors},
           "w_sel": {r["id"]: r.get("w_cand") for r in rows if r.get("etapa") in ("seleccion", "conjunto")},
           "w_sel_base": {r["id"]: r.get("w_base") for r in rows if r.get("etapa") in ("seleccion", "conjunto")}}
    (HERE / "finalistas.json").write_text(json.dumps(fin, indent=2))
    with pl.Config(tbl_rows=60, tbl_cols=12, fmt_str_lengths=40):
        print(res.select(["etapa", "id", "delta_dev", "se_dev", "p", "p_holm", "delta_logloss", "w_base", "w_cand",
                          "delta_rmse_total", "delta_rmse_margen", "decision"]))
    print("sobrevivientes:", survivors)


if __name__ == "__main__":
    main()
