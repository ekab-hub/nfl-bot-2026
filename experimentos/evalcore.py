"""Nucleo de evaluacion: GLM Poisson (con offset), walk-forward, metricas pareadas
con SE por bloques de semana, blend con Elo, NB y Poisson bivariado."""
import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import skellam, poisson, nbinom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DATA = Path(__file__).parent / "data"
BASE = ["team_rush_epa", "team_qb_value", "opp_def_epa_allowed", "is_home", "div_game"]
W_GRID = np.round(np.arange(0, 0.6001, 0.05), 2)
ALPHA_EVAL = 1e-4
ALPHA_PROD = 1.0
TRAIN_START = 2002
WEATHER = ["wind_obs", "cold_obs"]


# ------------------------------------------------------------------ GLM
def fit_poisson(X, y, alpha, offset=None):
    """Mismo objetivo que sklearn PoissonRegressor: mean(dev)/2 + alpha/2*||w||^2
    (intercepto sin penalizar), con offset opcional."""
    n, p = X.shape
    off = np.zeros(n) if offset is None else offset
    b0 = np.zeros(p + 1); b0[0] = np.log(y.mean()) - off.mean()

    def f(b):
        eta = b[0] + X @ b[1:] + off
        mu = np.exp(eta)
        loss = np.mean(mu - y * eta) + 0.5 * alpha * b[1:] @ b[1:]
        r = (mu - y) / n
        g = np.concatenate([[r.sum()], X.T @ r + alpha * b[1:]])
        return loss, g
    res = minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-10, "ftol": 1e-14})
    return res.x


def predict(b, X, offset=None):
    return np.exp(b[0] + X @ b[1:] + (0 if offset is None else offset))


def poisson_dev(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def walk_forward(df, feats, test_seasons, train_start=TRAIN_START, spec="eval", offset_col=None):
    """Devuelve las filas de prueba con mu fuera de muestra (y mu de entrenamiento por fold
    para estimar dispersion/correlacion)."""
    out, train_fits = [], {}
    for s in test_seasons:
        tr = df.filter(pl.col("season").is_between(train_start, s - 1))
        te = df.filter(pl.col("season") == s)
        Xtr = tr.select(feats).to_numpy().astype(float) if feats else np.zeros((tr.height, 0))
        Xte = te.select(feats).to_numpy().astype(float) if feats else np.zeros((te.height, 0))
        # imputacion de clima faltante (aire libre sin dato): mediana de entrenamiento
        for i, f in enumerate(feats):
            if f in WEATHER:
                med = np.nanmedian(Xtr[:, i][tr["indoor"].to_numpy() == 0])
                Xtr[:, i] = np.where(np.isnan(Xtr[:, i]), med, Xtr[:, i])
                Xte[:, i] = np.where(np.isnan(Xte[:, i]), med, Xte[:, i])
        assert not np.isnan(Xtr).any() and not np.isnan(Xte).any(), (feats, s)
        if spec == "eval":
            m, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
            Xtr, Xte = (Xtr - m) / sd, (Xte - m) / sd
            alpha = ALPHA_EVAL
        else:
            alpha = ALPHA_PROD
        otr = tr[offset_col].to_numpy() if offset_col else None
        ote = te[offset_col].to_numpy() if offset_col else None
        ytr = tr["points"].to_numpy().astype(float)
        b = fit_poisson(Xtr, ytr, alpha, otr)
        train_fits[s] = (tr.select(["game_id", "is_home", "points"]).with_columns(
            pl.Series("mu", predict(b, Xtr, otr))))
        out.append(te.select(["season", "week", "game_id", "team", "is_home", "points"]).with_columns(
            pl.Series("mu", predict(b, Xte, ote))))
    return pl.concat(out), train_fits


# ------------------------------------------------------------------ estadistica pareada
def norm_p(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def paired(d, clusters):
    """Media de d con SE cluster-robusto por bloques (temporada-semana)."""
    d = np.asarray(d, float); N = len(d); dbar = d.mean()
    cl = pl.DataFrame({"c": clusters, "d": d}).group_by("c").agg(pl.col("d").sum().alias("D"), pl.len().alias("n"))
    se = np.sqrt(((cl["D"].to_numpy() - cl["n"].to_numpy() * dbar) ** 2).sum()) / N
    z = dbar / se if se > 0 else 0.0
    return dbar, se, z, norm_p(z)


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i]); adj[i] = min(running, 1.0)
    return adj


def clusters_of(df):
    return (df["season"].cast(pl.Utf8) + "_" + df["week"].cast(pl.Utf8)).to_list()


# ------------------------------------------------------------------ nivel partido / blend
def to_games(pred):
    h = pred.filter(pl.col("is_home") == 1).select(["season", "week", "game_id", pl.col("team").alias("home_team"),
                                                    pl.col("points").alias("hp"), pl.col("mu").alias("lh")])
    a = pred.filter(pl.col("is_home") == 0).select(["game_id", pl.col("team").alias("away_team"),
                                                    pl.col("points").alias("ap"), pl.col("mu").alias("la")])
    return h.join(a, on="game_id").sort("game_id")


def p_home_skellam(lh, la, l3=0.0):
    return skellam.sf(0, lh - l3, la - l3)


_ELO = None
def elo_probs():
    """Elo de produccion con parametros EXPLICITOS (k=20, ha=35, rtm=0.40), historial 1999+."""
    global _ELO
    if _ELO is None:
        cache = DATA / "elo_k20_ha35_rtm040.parquet"
        if cache.exists():
            _ELO = pl.read_parquet(cache)
        else:
            from motor_elo_v1 import run_elo_over_seasons
            preds, _ = run_elo_over_seasons(list(range(1999, 2026)), k=20, home_advantage=35, regression_to_mean=0.40)
            _ELO = preds.filter(pl.col("game_type") == "REG").select(
                ["season", "week", "home_team", "away_team", pl.col("model_prob_home").alias("p_elo")])
            _ELO.write_parquet(cache)
    return _ELO


def attach_elo(games):
    g = games.join(elo_probs().with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)),
                   on=["season", "week", "home_team", "away_team"], how="left")
    assert g["p_elo"].null_count() == 0, "partidos sin Elo"
    return g


def logloss_vec(p, y):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def blend_eval(games, p_pois, w=None):
    """Log-loss del blend (sin empates). Si w es None, se optimiza en la grilla."""
    y = (games["hp"].to_numpy() > games["ap"].to_numpy()).astype(float)
    keep = games["hp"].to_numpy() != games["ap"].to_numpy()
    pe = games["p_elo"].to_numpy()
    losses = {wi: logloss_vec(wi * p_pois + (1 - wi) * pe, y)[keep].mean() for wi in W_GRID}
    w_opt = min(losses, key=losses.get)
    w_use = w_opt if w is None else w
    p = w_use * p_pois + (1 - w_use) * pe
    per = logloss_vec(p, y)
    acc = np.mean(((p > 0.5) == (y == 1))[keep])
    return {"w": float(w_use), "w_opt": float(w_opt), "logloss": float(per[keep].mean()), "acc": float(acc),
            "per_game": per, "keep": keep}


# ------------------------------------------------------------------ NB y bivariado
def fit_nb_r(y, mu):
    def nll(logr):
        r = np.exp(logr)
        return -nbinom.logpmf(y, r, r / (r + mu)).sum()
    res = minimize_scalar(nll, bounds=(-2, 8), method="bounded")
    return float(np.exp(res.x))


def nb_nll(y, mu, r):
    return -nbinom.logpmf(y, r, r / (r + mu))


def nb_p_home(lh, la, r, maxp=100):
    k = np.arange(maxp)
    out = np.empty(len(lh))
    for i, (a, b) in enumerate(zip(lh, la)):
        ph = nbinom.pmf(k, r, r / (r + a)); pa = nbinom.pmf(k, r, r / (r + b))
        cdf_a = np.cumsum(pa)
        out[i] = (ph[1:] * cdf_a[:-1]).sum()   # P(H > A)
    return out


def bivar_nll(h, a, lh, la, l3):
    """-log P(H=h, A=a) con H = X1+X3, A = X2+X3, X3 ~ Poi(l3)."""
    if l3 <= 0:
        return -(poisson.logpmf(h, lh) + poisson.logpmf(a, la))
    k = np.arange(0, 80)[None, :]
    h_ = np.asarray(h)[:, None]; a_ = np.asarray(a)[:, None]
    p = (poisson.pmf(h_ - k, (lh - l3)[:, None]) * poisson.pmf(a_ - k, (la - l3)[:, None])
         * poisson.pmf(k, l3)).sum(axis=1)
    return -np.log(np.maximum(p, 1e-300))


def fit_l3(h, a, lh, la, grid=np.arange(0, 6.01, 0.25)):
    best, bl = None, np.inf
    cap = np.minimum(lh, la)
    for l3 in grid:
        if l3 >= cap.min():
            break
        v = bivar_nll(h, a, lh, la, l3).sum()
        if v < bl:
            best, bl = l3, v
    return float(best)
