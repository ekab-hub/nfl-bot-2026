"""
Test de leakage explicito por feature (PROTOCOLO.md seccion 6).
1) Recalcula las features de 40+ temporada-semanas con datos TRUNCADOS a lo que se
   conocia antes de esa semana y exige igualdad con las features calculadas con todo.
2) Reporte de lesiones: date_modified del reporte usado vs fecha del partido.
Escribe data/leakage_result.csv (feature, n_comparadas, n_distintas, max_diff, pasa).
"""
import random
import numpy as np
import polars as pl
import features as F

FEATS = ["team_rush_epa", "opp_def_epa_allowed", "team_qb_value", "team_pass_epa", "team_success",
         "opp_def_pass_epa", "opp_def_rush_epa", "team_giveaway", "opp_takeaway", "team_sack_allowed",
         "opp_sack_rate", "team_pressure_allowed", "opp_pressure_rate", "team_explosive",
         "opp_explosive_allowed", "team_st_epa", "opp_st_epa", "team_pace", "opp_pace", "qb_value_live",
         "backup_qb", "ol_out", "wr1_out", "opp_dl_out", "opp_cb1_out", "indoor", "turf", "log_implied"]


def truncate(raw, s, w):
    t = dict(raw)
    before = (pl.col("season") < s) | ((pl.col("season") == s) & (pl.col("week") < w))
    t["pbp"] = raw["pbp"].filter(before)
    gids = t["pbp"]["game_id"].unique()
    t["participation"] = raw["participation"].filter(pl.col("nflverse_game_id").is_in(gids.implode()))
    sch = raw["schedules"].filter((pl.col("season") < s) | ((pl.col("season") == s) & (pl.col("week") <= w)))
    cur = (pl.col("season") == s) & (pl.col("week") == w)
    t["schedules"] = sch.with_columns([pl.when(cur).then(None).otherwise(pl.col(c)).alias(c)
                                       for c in ["home_score", "away_score", "result", "total"]])
    t["injuries"] = raw["injuries"].filter((pl.col("season") < s) | ((pl.col("season") == s) & (pl.col("week") <= w)))
    t["snap_counts"] = raw["snap_counts"].filter(before)
    t["draft_picks"] = raw["draft_picks"].filter(pl.col("season") <= s)
    return t


def main():
    raw = F.load_raw()
    full = pl.read_parquet(F.DATA / "features.parquet")
    random.seed(20261006)
    pairs = set()
    for s in random.sample(range(2003, 2026), 20):
        pairs.add((s, 1))
    while len(pairs) < 48:
        s = random.randint(2003, 2025); w = random.randint(2, 17)
        pairs.add((s, w))
    pairs = sorted(pairs)
    stats = {f: [0, 0, 0.0] for f in FEATS}
    for s, w in pairs:
        trunc = F.build_features(truncate(raw, s, w), seasons=[s]).filter(pl.col("week") == w)
        ref = full.filter((pl.col("season") == s) & (pl.col("week") == w))
        j = ref.select(["game_id", "team"] + FEATS).join(trunc.select(["game_id", "team"] + FEATS),
                                                         on=["game_id", "team"], suffix="_t")
        assert j.height == ref.height, (s, w, j.height, ref.height)
        for f in FEATS:
            a = j[f].cast(pl.Float64).to_numpy(); b = j[f + "_t"].cast(pl.Float64).to_numpy()
            both_nan = np.isnan(a) & np.isnan(b)
            d = np.where(both_nan, 0.0, np.abs(a - b))
            d = np.nan_to_num(d, nan=np.inf)
            stats[f][0] += len(d); stats[f][1] += int((d > 1e-9).sum()); stats[f][2] = max(stats[f][2], float(d.max()))
        print(s, w, "ok")
    rows = [{"feature": f, "n_comparadas": v[0], "n_distintas": v[1], "max_diff": v[2], "pasa": v[1] == 0}
            for f, v in stats.items()]
    # clima observado: no es pre-kickoff por construccion
    rows += [{"feature": f, "n_comparadas": 0, "n_distintas": 0, "max_diff": 0.0, "pasa": False}
             for f in ["wind_obs", "cold_obs"]]
    res = pl.DataFrame(rows)

    # 2) fecha del reporte de lesiones vs fecha del partido
    inj = raw["injuries"].filter((pl.col("game_type") == "REG") & pl.col("report_status").is_in(["Out", "Doubtful"]))
    tg = F.team_games(raw["schedules"]).select(["season", "week", "team", "gameday"])
    ij = inj.join(tg.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)),
                  on=["season", "week", "team"], how="inner")
    ij = ij.with_columns(pl.col("date_modified").dt.date().alias("d"), pl.col("gameday").str.to_date().alias("g"))
    late = ij.with_columns((pl.col("d") > pl.col("g")).alias("late"))
    by = late.group_by("season").agg(pl.col("late").mean().alias("frac_reporte_posterior"), pl.len()).sort("season")
    res.write_csv(F.DATA / "leakage_result.csv")
    by.write_csv(F.DATA / "leakage_injury_dates.csv")
    with pl.Config(tbl_rows=40):
        print(res); print(by)


if __name__ == "__main__":
    main()
