"""Descarga y cachea en experimentos/data/ todo lo que usan los experimentos."""
from pathlib import Path
import nflreadpy as nfl
import polars as pl

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

PBP_COLS = ["game_id", "season", "season_type", "week", "game_date", "posteam", "defteam",
            "play_type", "epa", "qb_epa", "success", "qb_dropback", "sack", "interception",
            "fumble_lost", "yards_gained", "rush_attempt", "pass_attempt",
            "passer_player_id", "passer_player_name", "special_teams_play", "play_id",
            "home_team", "away_team"]


def main():
    if not (DATA / "pbp.parquet").exists():
        parts = []
        for s in range(2001, 2026):
            p = nfl.load_pbp(seasons=[s])
            parts.append(p.select([c for c in PBP_COLS if c in p.columns]))
            print("pbp", s, p.height)
        pl.concat(parts, how="diagonal_relaxed").write_parquet(DATA / "pbp.parquet")

    nfl.load_schedules().write_parquet(DATA / "schedules.parquet")
    nfl.load_draft_picks().write_parquet(DATA / "draft_picks.parquet")
    nfl.load_players().write_parquet(DATA / "players.parquet")
    nfl.load_injuries(seasons=list(range(2009, 2026))).write_parquet(DATA / "injuries.parquet")
    nfl.load_snap_counts(seasons=list(range(2012, 2026))).write_parquet(DATA / "snap_counts.parquet")

    parts = []
    for s in range(2016, 2026):
        d = nfl.load_participation(seasons=[s])
        parts.append(d.select(["nflverse_game_id", "play_id", "possession_team", "was_pressure"])
                      .with_columns(pl.lit(s).alias("season")))
    pl.concat(parts, how="diagonal_relaxed").write_parquet(DATA / "participation.parquet")
    print("ok")


if __name__ == "__main__":
    main()
