"""
Calibracion spread -> P(local gana) con nflreadpy load_schedules 2010-2025 (REG).
- Verifica la convencion de signo de spread_line.
- Ajusta: normal (sigma libre), normal (sigma + sesgo), logistica (pendiente),
  logistica (pendiente + sesgo); compara contra normal sigma=13.5.
- Comparacion fuera de muestra: leave-one-season-out (LOSO) log-loss.
- Empates: excluidos del ajuste (se reporta cuantos).
Escribe data/mercado_calibracion.json y data/mercado_calibracion_tabla.csv.
"""
import json
from pathlib import Path
import numpy as np
import polars as pl
import nflreadpy as nfl
from scipy.optimize import minimize
from scipy.stats import norm

DATA = Path(__file__).parent / "data"


def load():
    s = nfl.load_schedules(seasons=list(range(2010, 2026)))
    s = s.filter((pl.col("game_type") == "REG") & pl.col("home_score").is_not_null() & pl.col("spread_line").is_not_null())
    return s.with_columns((pl.col("home_score") - pl.col("away_score")).alias("margin"))


def p_model(name, params, x):
    if name == "normal_13.5":
        return norm.cdf(x / 13.5)
    if name == "normal":
        return norm.cdf(x / params[0])
    if name == "normal_sesgo":
        return norm.cdf((x + params[1]) / params[0])
    if name == "logistica":
        return 1 / (1 + np.exp(-params[0] * x))
    if name == "logistica_sesgo":
        return 1 / (1 + np.exp(-(params[0] * x + params[1])))


INIT = {"normal": [13.5], "normal_sesgo": [13.5, 0.0], "logistica": [0.13], "logistica_sesgo": [0.13, 0.0]}


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(name, x, y):
    if name == "normal_13.5":
        return []
    res = minimize(lambda th: ll(p_model(name, th, x), y).mean(), INIT[name], method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000})
    return list(res.x)


def main():
    s = load()
    # 1) convencion de signo
    corr = np.corrcoef(s["spread_line"].to_numpy(), s["margin"].to_numpy())[0, 1]
    fav_home = s.filter(pl.col("spread_line") > 0)
    sign = {"corr_spread_margen": corr,
            "frac_local_gana_si_spread>0": float((fav_home["margin"] > 0).mean()),
            "frac_local_gana_si_spread<0": float((s.filter(pl.col("spread_line") < 0)["margin"] > 0).mean()),
            "margen_medio_por_punto_de_spread": float(np.polyfit(s["spread_line"].to_numpy(), s["margin"].to_numpy(), 1)[0])}
    print("signo:", sign)
    assert corr > 0, "spread_line no es margen esperado del local"

    ties = s.filter(pl.col("margin") == 0).height
    d = s.filter(pl.col("margin") != 0)
    x = d["spread_line"].to_numpy().astype(float); y = (d["margin"].to_numpy() > 0).astype(float)
    seasons = d["season"].to_numpy()
    sd_resid = float(np.std(s["margin"].to_numpy() - s["spread_line"].to_numpy()))

    out = {"signo": sign, "n_partidos": s.height, "empates_excluidos": ties, "sd_margen_menos_spread": sd_resid,
           "modelos": {}}
    preds_loso = {}
    for name in ["normal_13.5", "normal", "normal_sesgo", "logistica", "logistica_sesgo"]:
        full = fit(name, x, y)
        p_cv = np.empty(len(y))
        for sz in np.unique(seasons):
            m = seasons == sz
            th = fit(name, x[~m], y[~m])
            p_cv[m] = p_model(name, th, x[m])
        preds_loso[name] = p_cv
        out["modelos"][name] = {"params": full, "logloss_insample": float(ll(p_model(name, full, x), y).mean()),
                                "logloss_loso": float(ll(p_cv, y).mean()),
                                "brier_loso": float(np.mean((p_cv - y) ** 2))}
    # diferencias pareadas LOSO vs normal 13.5 (SE por bloques de semana)
    cl = (d["season"].cast(pl.Utf8) + "_" + d["week"].cast(pl.Utf8)).to_numpy()
    base = ll(preds_loso["normal_13.5"], y)
    for name, p in preds_loso.items():
        dd = ll(p, y) - base
        df = pl.DataFrame({"c": cl, "d": dd}).group_by("c").agg(pl.col("d").sum().alias("D"), pl.len().alias("n"))
        se = np.sqrt(((df["D"].to_numpy() - df["n"].to_numpy() * dd.mean()) ** 2).sum()) / len(dd)
        out["modelos"][name]["delta_vs_13.5"] = float(dd.mean()); out["modelos"][name]["se_delta"] = float(se)
    for k, v in out["modelos"].items():
        print(f"{k:16s} params={np.round(v['params'], 4)}  LL_in={v['logloss_insample']:.5f}  "
              f"LL_loso={v['logloss_loso']:.5f}  delta_vs_13.5={v['delta_vs_13.5']:+.5f} (SE {v['se_delta']:.5f})")

    # tabla de calibracion (tramos de spread) con el modelo elegido y con 13.5
    sigma = out["modelos"]["normal"]["params"][0]
    bins = [-30, -10.5, -7.5, -4.5, -2.5, -0.5, 0.5, 2.5, 4.5, 7.5, 10.5, 30]
    labels = ["<=-11", "-10 a -8", "-7 a -5", "-4 a -3", "-2 a -1", "pick (0)", "+1 a +2", "+3 a +4", "+5 a +7",
              "+8 a +10", ">=+11"]
    t = d.with_columns(pl.col("spread_line").cut(bins[1:-1], labels=labels).alias("tramo"),
                       pl.Series("y", y), pl.Series("p_fit", norm.cdf(x / sigma)), pl.Series("p_135", norm.cdf(x / 13.5)))
    tab = t.group_by("tramo").agg(pl.len().alias("n"), pl.col("spread_line").mean().alias("spread_medio"),
                                  pl.col("p_fit").mean().alias("p_pred_sigma_fit"), pl.col("p_135").mean().alias("p_pred_13.5"),
                                  pl.col("y").mean().alias("frec_real")).with_columns(
        (pl.col("frec_real") * (1 - pl.col("frec_real")) / pl.col("n")).sqrt().alias("se_frec")).sort("spread_medio")
    with pl.Config(tbl_rows=20):
        print(tab)
    tab.write_csv(DATA / "mercado_calibracion_tabla.csv")
    (DATA / "mercado_calibracion.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
