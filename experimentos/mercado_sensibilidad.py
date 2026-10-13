"""Sensibilidad del peso optimo a lineas menos informadas que el cierre:
spread_martes ~= spread_cierre + N(0, tau^2) (aproximacion; ruido independiente del resultado).
sigma de spread_to_prob se mantiene fijo (el de cierre), como hara el pipeline. 50 sorteos por tau."""
import numpy as np, polars as pl
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import evalcore as E
import mercado_peso as M
import seleccion as S

SIGMA = 11.127   # ajuste 2010-2025 (mercado_calibracion.py)
sel = M.games_with_market(list(range(2010, 2023)), S.DF)
y = (sel["hp"] > sel["ap"]).to_numpy().astype(float); pb = sel["p_base"].to_numpy(); x = sel["spread_line"].to_numpy()
rng = np.random.default_rng(20261006)
rows = []
for tau in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]:
    ws, ll20, llbest, ll_mkt = [], [], [], []
    for _ in range(1 if tau == 0 else 50):
        pm = norm.cdf((x + rng.normal(0, tau, len(x))) / SIGMA)
        f = lambda w: E.logloss_vec((1 - w) * pb + w * pm, y).mean()
        w_opt = minimize_scalar(f, bounds=(0, 1), method="bounded").x
        ws.append(w_opt); ll20.append(f(0.2) - f(0)); llbest.append(f(min(w_opt, 0.6)) - f(0)); ll_mkt.append(f(1.0) - f(0))
    rows.append({"tau_puntos": tau, "w_optimo_medio": np.mean(ws), "w_p10": np.percentile(ws, 10), "w_p90": np.percentile(ws, 90),
                 "delta_LL_w0.20": np.mean(ll20), "delta_LL_w_opt(<=0.6)": np.mean(llbest), "delta_LL_solo_mercado": np.mean(ll_mkt)})
t = pl.DataFrame(rows); t.write_csv(E.DATA / "mercado_sensibilidad.csv")
with pl.Config(tbl_cols=10, tbl_width_chars=200): print(t)
