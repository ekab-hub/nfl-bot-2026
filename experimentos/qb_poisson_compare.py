"""Chequeos de Parte 1: (a) perdida de spread por tramos de caida (muestra ii, 2010-2022, un solo
equipo con cambio); (b) efecto implicito del Poisson semanal (JOINT_PROD 2002-2022) para elite->suplente."""
import json
import numpy as np
import polars as pl
import evalcore as E
import qb_mercado as Q

d = pl.read_parquet(E.DATA / "qb_mercado_juegos.parquet")
out = json.loads((E.DATA / "qb_mercado.json").read_text())
a = [r for r in out["seleccion"] if r["spec"] == "ii"][0]["a"]
s2 = Q.samples(d.filter(pl.col("season") <= 2022))["ii"]
one = s2.filter((pl.col("home_caida") != 0) ^ (pl.col("away_caida") != 0))
# perdida de spread del equipo que cambia (positivo = el mercado lo castiga), neta del intercepto
t = one.with_columns(
    pl.when(pl.col("home_caida") != 0).then(pl.col("home_caida")).otherwise(pl.col("away_caida")).alias("caida"),
    pl.when(pl.col("home_caida") != 0).then(-(pl.col("y") - a)).otherwise(pl.col("y") - a).alias("perdida_spread"))
t = t.with_columns(pl.col("caida").cut([-0.05, 0.05, 0.15, 0.25], labels=["< -0.05 (mejora)", "-0.05 a 0.05", "0.05 a 0.15", "0.15 a 0.25", "> 0.25"]).alias("tramo"))
tab = t.group_by("tramo").agg(pl.len().alias("n"), pl.col("caida").mean().alias("caida_media"),
                              pl.col("perdida_spread").mean().alias("perdida_media"),
                              (pl.col("perdida_spread").std() / pl.len().sqrt()).alias("se")).sort("caida_media")
print(tab); tab.write_csv(E.DATA / "qb_mercado_tramos.csv")

# Poisson semanal JOINT_PROD
fin = json.loads((E.DATA.parent / "finalistas.json").read_text())["joints"]["JOINT_PROD"]
df = pl.read_parquet(E.DATA / "features.parquet").filter(pl.col("season").is_between(2002, 2022))
X = df.select(fin["feats"]).to_numpy().astype(float); m, sd = X.mean(0), X.std(0)
b = E.fit_poisson((X - m) / sd, df["points"].to_numpy().astype(float), 1e-4)
raw = dict(zip(fin["feats"], b[1:] / sd))
mu = E.predict(b, (X - m) / sd)
dv = out["elite_vs_suplente"]["dv"]
eff_full = float(np.mean(mu * (np.exp(-raw["qb_value_live"] * dv + raw["backup_qb"]) - 1)))
eff_val = float(np.mean(mu * (np.exp(-raw["qb_value_live"] * dv) - 1)))
eff_bk = float(np.mean(mu * (np.exp(raw["backup_qb"]) - 1)))
mkt = out["elite_vs_suplente"]["puntos"]
res = {"beta_qb_value_live": raw["qb_value_live"], "beta_backup_qb": raw["backup_qb"], "lambda_medio": float(mu.mean()),
       "dv": dv, "poisson_margen_total": eff_full, "poisson_solo_valor": eff_val, "poisson_solo_backup": eff_bk,
       "en_blend_x0.45": 0.45 * eff_full, "mercado": -mkt,
       "poisson_por_unidad_valor": float(np.mean(mu) * raw["qb_value_live"])}
print(json.dumps(res, indent=2)); (E.DATA / "qb_poisson_compare.json").write_text(json.dumps(res, indent=2))
