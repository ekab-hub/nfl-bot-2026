"""
Peso del mercado sobre el blend base (0.45 Poisson JOINT_PROD + 0.55 Elo):
p_final = (1-w)*base + w*Phi(spread/sigma), w en 0..0.6.
Seleccion 2010-2022 decide; 2023-2025 solo informativo (holdout ya usado).
sigma se ajusta solo con 2010-2022 para la busqueda en seleccion.
"""
import json
import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import evalcore as E
import seleccion as S

W = np.round(np.arange(0, 0.6001, 0.05), 2)
POIS_W = 0.45


def games_with_market(seasons, df):
    fin = json.loads((E.DATA.parent / "finalistas.json").read_text())["joints"]["JOINT_PROD"]
    pred, _ = S.run(fin["feats"], fin["start"], seasons=seasons, df=df)
    g, p_pois = S.game_level(pred)
    spread = df.filter(pl.col("is_home") == 1).select(["game_id", "spread_line"])
    g = g.with_columns(pl.Series("p_pois", p_pois)).join(spread, on="game_id", how="left")
    assert g["spread_line"].null_count() == 0
    g = g.with_columns((POIS_W * pl.col("p_pois") + (1 - POIS_W) * pl.col("p_elo")).alias("p_base"))
    return g.filter(pl.col("hp") != pl.col("ap"))   # empates fuera


def fit_sigma(g):
    x = g["spread_line"].to_numpy(); y = (g["hp"] > g["ap"]).to_numpy().astype(float)
    return minimize_scalar(lambda s: E.logloss_vec(norm.cdf(x / s), y).mean(), bounds=(5, 25), method="bounded").x


def curve(g, sigma):
    y = (g["hp"] > g["ap"]).to_numpy().astype(float)
    pb = g["p_base"].to_numpy(); pm = norm.cdf(g["spread_line"].to_numpy() / sigma)
    cl = E.clusters_of(g)
    l0 = E.logloss_vec(pb, y)
    rows = []
    for w in list(W) + [1.0]:
        lw = E.logloss_vec((1 - w) * pb + w * pm, y)
        d, se, z, p = E.paired(lw - l0, cl)
        acc = np.mean(((1 - w) * pb + w * pm > 0.5) == (y == 1))
        rows.append({"w": float(w), "logloss": float(lw.mean()), "delta_vs_w0": d, "se": se, "acc": float(acc)})
    return pl.DataFrame(rows)


def main():
    sel = games_with_market(list(range(2010, 2023)), S.DF)
    sigma_sel = fit_sigma(sel)
    c_sel = curve(sel, sigma_sel)
    ho = games_with_market([2023, 2024, 2025], S.df_all)
    c_ho = curve(ho, sigma_sel)
    # w optimo continuo (sin restringir a la grilla) como informacion
    y = (sel["hp"] > sel["ap"]).to_numpy().astype(float)
    pb = sel["p_base"].to_numpy(); pm = norm.cdf(sel["spread_line"].to_numpy() / sigma_sel)
    w_cont = minimize_scalar(lambda w: E.logloss_vec((1 - w) * pb + w * pm, y).mean(), bounds=(0, 1), method="bounded").x
    # optimo vs 0.20 pareado
    l20 = E.logloss_vec(0.8 * pb + 0.2 * pm, y)
    wbest = float(c_sel.filter(pl.col("w") <= 0.6).sort("logloss")["w"][0])
    lb = E.logloss_vec((1 - wbest) * pb + wbest * pm, y)
    d_best_20 = E.paired(lb - l20, E.clusters_of(sel))
    print(f"sigma (2010-2022) = {sigma_sel:.3f};  w continuo optimo (0-1) = {w_cont:.3f}")
    with pl.Config(tbl_rows=20):
        print("SELECCION 2010-2022, n =", sel.height); print(c_sel)
        print("2023-2025 (informativo), n =", ho.height); print(c_ho)
    print(f"w={wbest} vs w=0.20 en seleccion: delta={d_best_20[0]:+.5f} SE={d_best_20[1]:.5f}")
    out = {"sigma_sel": sigma_sel, "w_continuo": w_cont, "w_grilla": wbest,
           "delta_best_vs_020": d_best_20[:2], "n_sel": sel.height, "n_ho": ho.height}
    (E.DATA / "mercado_peso.json").write_text(json.dumps(out, indent=2, default=float))
    c_sel.write_csv(E.DATA / "mercado_peso_seleccion.csv"); c_ho.write_csv(E.DATA / "mercado_peso_2023_2025.csv")


if __name__ == "__main__":
    main()
