"""Compara las features baseline reimplementadas contra poisson_features.build_game_features
(produccion, solo lectura) para una temporada."""
import sys
from pathlib import Path
import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poisson_features import build_game_features  # produccion (solo lectura)
import features as F

SEASON = 2023
START = 2018

prod = build_game_features(seasons_to_build=[SEASON], pbp_history_start=START) \
    .filter(pl.col("game_id").str.contains("_(0[1-9]|1[0-8])_"))  # solo REG
raw = F.load_raw()
raw_r = dict(raw)
raw_r["pbp"] = raw["pbp"].filter(pl.col("season") >= START)
mine = F.build_features(raw_r, seasons=[SEASON])

j = prod.join(mine.select(["game_id", "team", "team_rush_epa", "opp_def_epa_allowed", "team_qb_value", "points"]),
              on=["game_id", "team"], how="inner", suffix="_mine")
print("filas prod", prod.height, "mine", mine.height, "cruzadas", j.height)
for c in ["team_rush_epa", "opp_def_epa_allowed", "team_qb_value", "points_scored"]:
    m = c + "_mine" if c != "points_scored" else "points"
    d = np.abs(j[c].to_numpy() - j[m].to_numpy())
    print(f"{c:22s} max|diff|={d.max():.2e}  frac>1e-9={np.mean(d > 1e-9):.3f}")
bad = j.filter((pl.col("team_qb_value") - pl.col("team_qb_value_mine")).abs() > 1e-9)
print("QB distintos (esperado: solo novatos por la correccion de cubeta):",
      bad.select(["week", "team", "team_qb_value", "team_qb_value_mine"]).unique(["team", "team_qb_value"]).head(20))
