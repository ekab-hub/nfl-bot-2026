import polars as pl
import nflreadpy as nfl
import weekly_pipeline as wp

SEASON, WEEK = 2026, 5
TEAMS = ["BAL", "ATL"]
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(250)

sch = nfl.load_schedules([SEASON]).filter(
    (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null())

for t in TEAMS:
    h = sch.filter(pl.col("home_team") == t).select(
        "week", pl.col("home_qb_id").alias("qb_id"), pl.col("home_qb_name").alias("qb"))
    a = sch.filter(pl.col("away_team") == t).select(
        "week", pl.col("away_qb_id").alias("qb_id"), pl.col("away_qb_name").alias("qb"))
    print(f"\n== {t}: titulares {SEASON} ==")
    print(pl.concat([h, a]).sort("week"))

print("\n== features de la semana", WEEK, "==")
try:
    out = wp.build_week_features(SEASON, WEEK)
    out = out[0] if isinstance(out, tuple) else out
    df = out if isinstance(out, pl.DataFrame) else pl.DataFrame(out)
    cols = [c for c in ["team", "side", "backup_qb", "team_qb_value", "qb_value_live",
                        "team_pass_epa", "team_rush_epa", "team_success",
                        "opp_def_epa_allowed", "is_home"] if c in df.columns]
    print(df.filter(pl.col("team").is_in(TEAMS)).select(cols))
except Exception as e:
    print("error:", type(e).__name__, e)