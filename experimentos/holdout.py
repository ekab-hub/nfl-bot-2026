"""
Etapa HOLDOUT (2023-2025). Se corre UNA sola vez (candado en data/HOLDOUT_LOCK).
Solo finalistas de finalistas.json. w del blend = el elegido en seleccion (no se
re-optimiza; el optimo de holdout se reporta como informacion).
Agrega filas etapa=holdout a resultados.csv y escribe la decision final.
"""
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl
import evalcore as E
import seleccion as S

HERE = Path(__file__).parent
LOCK = E.DATA / "HOLDOUT_LOCK"
DRY = os.environ.get("HOLDOUT_DRYRUN") == "1"   # prueba del codigo sobre seleccion, sin escribir
HO = [2020, 2021, 2022] if DRY else [2023, 2024, 2025]


def main():
    if LOCK.exists() and not DRY:
        raise SystemExit(f"Holdout ya fue usado ({LOCK.read_text().strip()}). No se vuelve a correr.")
    fin = json.loads((HERE / "finalistas.json").read_text())
    df = S.df_all
    assert df["season"].max() == 2025
    if DRY:
        df = df.filter(pl.col("season") <= 2022)
    wc, wb = fin["w_sel"], fin["w_sel_base"]
    rows = []

    def run(feats, start=E.TRAIN_START, spec="eval"):
        return S.run(feats, start, seasons=HO, spec=spec, df=df)

    # referencia: produccion (spec prod, w=0.25) y Elo solo
    pp, _ = run(E.BASE, spec="prod")
    g, p = S.game_level(pp)
    b = E.blend_eval(g, p, 0.25)
    y = (g["hp"] > g["ap"]).to_numpy().astype(float); keep = b["keep"]; pe = g["p_elo"].to_numpy()
    rows.append({"etapa": "holdout", "id": "PROD", "descripcion": "baseline spec produccion, blend w=0.25 fijo",
                 "temporadas_test": "2023-2025", "n_partidos": int(keep.sum()), "logloss_cand": b["logloss"],
                 "acc_cand": b["acc"], "w_cand": 0.25, "logloss_base": float(E.logloss_vec(pe, y)[keep].mean()),
                 "acc_base": float(np.mean(((pe > 0.5) == (y == 1))[keep])), "decision": "referencia (base = Elo solo)"})

    # S0
    pe_, fits = run(E.BASE)
    r = S.compare(pp, pe_, w_b=wb["S0"], w_c=wc["S0"]); r.update(id="S0"); rows.append(r)

    # factores individuales
    for cid, c in fin["candidates"].items():
        pb, _ = run(E.BASE, c["start"]); pc, _ = run(c["feats"], c["start"])
        r = S.compare(pb, pc, w_b=wb[cid], w_c=wc[cid]); r.update(id=cid); rows.append(r)

    # conjuntos
    for jid, jd in fin["joints"].items():
        pb, _ = run(E.BASE, jd["start"]); pc, _ = run(jd["feats"], jd["start"])
        r = S.compare(pb, pc, w_b=wb[jid], w_c=wc[jid]); r.update(id=jid); rows.append(r)

    # O1 (binomial negativa) si es finalista
    if "O1" in fin["otros_finalistas"]:
        j = pe_.sort(["game_id", "team"])
        rs = {s: E.fit_nb_r(f["points"].to_numpy(), f["mu"].to_numpy()) for s, f in fits.items()}
        r_vec = np.array([rs[s] for s in j["season"].to_list()])
        nll_p = -E.poisson.logpmf(j["points"].to_numpy(), j["mu"].to_numpy())
        nll_nb = E.nb_nll(j["points"].to_numpy(), j["mu"].to_numpy(), r_vec)
        def pfn_nb(g):
            out = np.empty(g.height)
            for s in set(g["season"].to_list()):
                m = (g["season"] == s).to_numpy()
                out[m] = E.nb_p_home(g["lh"].to_numpy()[m], g["la"].to_numpy()[m], rs[s])
            return out
        r = S.compare(j, j, metric_b=2 * nll_p, metric_c=2 * nll_nb, pfn_c=pfn_nb, w_b=wb["O1"], w_c=wc["O1"])
        r.update(id="O1"); rows.append(r)

    for r in rows:
        r["etapa"] = "holdout"

    # decision final
    res = pl.read_csv(HERE / "resultados.csv", infer_schema_length=None)
    sel = {r["id"]: r for r in res.filter(pl.col("etapa").is_in(["seleccion", "conjunto"])).to_dicts()}
    ho = {r["id"]: r for r in rows}
    final = {}
    for cid, h in ho.items():
        if cid == "PROD":
            continue
        s = sel[cid]
        a = bool(s.get("pasa_a")) if cid not in fin["joints"] else True
        b_ = h["delta_logloss"] < 0
        c_ = s["delta_logloss"] <= -0.001
        d_ = bool(s["leakage_ok"])
        fails = [k for k, ok in [("a", a), ("b", b_), ("c", c_), ("d", d_)] if not ok]
        final[cid] = "ACEPTADO" if not fails else "RECHAZADO: falla " + ",".join(f"({k})" for k in fails)
        h["decision"] = final[cid]
    res = res.with_columns(pl.col("id").replace_strict(final, default=None).alias("_f")).with_columns(
        pl.when(pl.col("_f").is_not_null() & pl.col("etapa").is_in(["seleccion", "conjunto"]))
        .then(pl.col("_f")).otherwise(pl.col("decision")).alias("decision")).drop("_f")
    out = pl.concat([res, pl.DataFrame(rows, infer_schema_length=None)], how="diagonal_relaxed")
    if DRY:
        print("DRY RUN OK", out.shape, final)
        return
    LOCK.write_text(f"holdout corrido {datetime.now().isoformat()}\n")
    out.write_csv(HERE / "resultados.csv")
    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=250):
        print(pl.DataFrame(rows, infer_schema_length=None).select(
            ["id", "delta_dev", "se_dev", "p", "w_base", "w_cand", "w_opt_base", "w_opt_cand", "logloss_base",
             "delta_logloss", "se_logloss", "delta_rmse_total", "delta_rmse_margen", "decision"]))
    print(final)


if __name__ == "__main__":
    main()
