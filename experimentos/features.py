"""
Features por equipo-partido (temporada regular) para el Poisson de puntos.

Reimplementa vectorizado el baseline de produccion (team_epa.py, poisson_features.py,
qb_adjustment.py) y agrega los candidatos de PROTOCOLO.md. Todas las funciones reciben
los datos crudos como argumento para que leakage_test.py pueda pasarles datos truncados.

Regla temporal: la feature de un partido de la semana `wk` de la temporada `s` solo
usa jugadas/partidos de semanas < wk de `s`, temporadas < s, y el reporte de lesiones
de la semana `wk` (publicado antes del partido).
"""
from pathlib import Path
import numpy as np
import polars as pl

DATA = Path(__file__).parent / "data"
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
DECAY = 0.85          # produccion: team_epa.weekly_epa_with_blend
BLEND_GAMES = 8       # produccion
QB_SEASON_DECAY = 0.6     # produccion: qb_career_value
QB_MIN_DROPBACKS = 100    # produccion
QB_LIVE_PRIOR_N = 150     # C4c (igual a full_shrinkage_dropbacks de produccion)
TURNOVER_SHRINK = 0.5     # C2a
FIRST_SEASON = 2002
AVAIL_START = 2013       # snap counts efectivos desde 2013
LAST_SEASON = 2025
OL_POS = {"T", "G", "C", "OT", "OG", "OL"}
DL_POS = {"DE", "DT", "NT", "DL"}   # ver Desviaciones: OLB no se distingue de LB en lesiones


def norm_team(col):
    return pl.col(col).replace(TEAM_MAP)


def load_raw():
    raw = {k: pl.read_parquet(DATA / f"{k}.parquet") for k in
           ["pbp", "schedules", "draft_picks", "players", "injuries", "snap_counts", "participation"]}
    raw["schedules"] = raw["schedules"].with_columns(norm_team("home_team"), norm_team("away_team"))
    raw["injuries"] = raw["injuries"].with_columns(norm_team("team"),
                                                   pl.col("season").cast(pl.Int32),
                                                   pl.col("week").cast(pl.Int32))
    raw["snap_counts"] = raw["snap_counts"].with_columns(norm_team("team"))
    raw["pbp"] = raw["pbp"].with_columns(norm_team("posteam"), norm_team("defteam"))
    return raw


# ---------------------------------------------------------------- calendario largo
def team_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Una fila por equipo-partido de temporada regular (incluye partidos sin jugar)."""
    s = schedules.filter((pl.col("game_type") == "REG") &
                         pl.col("season").is_between(FIRST_SEASON - 1, LAST_SEASON))
    common = ["season", "week", "game_id", "gameday", "div_game", "roof", "surface", "temp",
              "wind", "total_line", "spread_line"]
    home = s.select(common + [
        pl.col("home_team").alias("team"), pl.col("away_team").alias("opp"),
        pl.lit(1).alias("is_home"), pl.col("home_score").alias("points"),
        pl.col("away_score").alias("opp_points"), pl.col("home_qb_id").alias("qb_id"),
        pl.col("away_qb_id").alias("opp_qb_id")])
    away = s.select(common + [
        pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"),
        pl.lit(0).alias("is_home"), pl.col("away_score").alias("points"),
        pl.col("home_score").alias("opp_points"), pl.col("away_qb_id").alias("qb_id"),
        pl.col("home_qb_id").alias("opp_qb_id")])
    return pl.concat([home, away]).sort(["season", "week", "game_id", "is_home"])


# ---------------------------------------------------------------- maquinaria decay+blend
def decayed_blend(agg: pl.DataFrame, num: str, den: str, games: pl.DataFrame,
                  target: pl.DataFrame, name: str) -> pl.DataFrame:
    """
    agg: (season, week, team, num, den) -- sumas por equipo-semana (incluye playoffs).
    games: (season, week, team) semanas en que el equipo jugo (para games_played).
    target: (season, week, team) filas a calcular.
    valor = bw * sum_{w<wk} d^(wk-w) num / sum d^(wk-w) den + (1-bw) * prior,
    prior = num/den de la temporada anterior completa (fallback: media de liga),
    bw = min(games_played/8, 1). Identico a team_epa.weekly_epa_with_blend
    (ponderar cada jugada por d^(wk-w) == ponderar las sumas semanales).
    """
    out = []
    for s in sorted(target["season"].unique().to_list()):
        tgt = target.filter(pl.col("season") == s)
        teams = sorted(set(tgt["team"].to_list()))
        tix = {t: i for i, t in enumerate(teams)}
        W = int(max(tgt["week"].max(), 1)) + 1
        a = agg.filter((pl.col("season") == s) & (pl.col("week") < W))
        N = np.zeros((len(teams), W + 1)); D = np.zeros((len(teams), W + 1)); G = np.zeros((len(teams), W + 1))
        for t, w, n, d in a.select(["team", "week", num, den]).iter_rows():
            if t in tix:
                N[tix[t], w] += n; D[tix[t], w] += d
        for t, w in games.filter((pl.col("season") == s) & (pl.col("week") < W)).select(["team", "week"]).iter_rows():
            if t in tix:
                G[tix[t], w] = 1
        prev = agg.filter(pl.col("season") == s - 1)
        league = prev[num].sum() / prev[den].sum() if prev.height and prev[den].sum() > 0 else 0.0
        pt = prev.group_by("team").agg(pl.col(num).sum(), pl.col(den).sum())
        prior = np.full(len(teams), league)
        for t, n, d in pt.iter_rows():
            if t in tix and d > 0:
                prior[tix[t]] = n / d
        rows_t, rows_w, rows_v = [], [], []
        for wk in sorted(tgt["week"].unique().to_list()):
            w = np.arange(W + 1)
            wt = np.where(w < wk, DECAY ** (wk - w).astype(float), 0.0)
            wt[0] = 0.0
            sn = N @ wt; sd = D @ wt
            gp = G[:, :wk].sum(axis=1)
            bw = np.minimum(gp / BLEND_GAMES, 1.0)
            cur = np.divide(sn, sd, out=np.zeros_like(sn), where=sd > 0)
            bw = np.where(sd > 0, bw, 0.0)
            val = bw * cur + (1 - bw) * prior
            rows_t += teams; rows_w += [wk] * len(teams); rows_v += val.tolist()
        out.append(pl.DataFrame({"season": [s] * len(rows_t), "week": rows_w, "team": rows_t, name: rows_v},
                                schema={"season": pl.Int32, "week": pl.Int32, "team": pl.Utf8, name: pl.Float64}))
    res = pl.concat(out)
    return target.join(res, on=["season", "week", "team"], how="left")


def league_prior(agg, num, den, seasons):
    """Media de liga de la temporada anterior (conocida antes de la temporada)."""
    rows = []
    for s in seasons:
        prev = agg.filter(pl.col("season") == s - 1)
        rows.append({"season": s, "league": prev[num].sum() / prev[den].sum() if prev.height else 0.0})
    return pl.DataFrame(rows, schema={"season": pl.Int32, "league": pl.Float64})


# ---------------------------------------------------------------- agregados de pbp
def pbp_aggregates(pbp: pl.DataFrame, participation: pl.DataFrame):
    plays = pbp.filter(pl.col("play_type").is_in(["run", "pass"]) & pl.col("epa").is_not_null()
                       & pl.col("posteam").is_not_null())
    db = (pl.col("qb_dropback") == 1)
    plays = plays.with_columns([
        pl.lit(1.0).alias("n"),
        (pl.col("play_type") == "run").cast(pl.Float64).alias("is_run"),
        db.cast(pl.Float64).alias("is_db"),
        (~db).cast(pl.Float64).alias("is_nondb"),
        pl.col("success").fill_null(0).cast(pl.Float64).alias("succ"),
        pl.col("sack").fill_null(0).cast(pl.Float64).alias("sk"),
        (pl.col("interception").fill_null(0) + pl.col("fumble_lost").fill_null(0)).cast(pl.Float64).alias("gv"),
        ((db & (pl.col("yards_gained") >= 20)) | (~db & (pl.col("yards_gained") >= 10)))
        .fill_null(False).cast(pl.Float64).alias("expl"),
    ]).with_columns([
        (pl.col("epa") * pl.col("is_run")).alias("epa_run"),
        (pl.col("epa") * pl.col("is_db")).alias("epa_db"),
        (pl.col("epa") * pl.col("is_nondb")).alias("epa_nondb"),
    ])
    pa = participation.select(["nflverse_game_id", "play_id", "was_pressure"]).unique(["nflverse_game_id", "play_id"])
    plays = plays.join(pa, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left")
    plays = plays.with_columns([
        (pl.col("is_db") * pl.col("was_pressure").is_not_null().cast(pl.Float64)).alias("pr_obs"),
        (pl.col("is_db") * pl.col("was_pressure").fill_null(False).cast(pl.Float64)).alias("pr"),
    ])
    cols = ["n", "epa", "is_run", "epa_run", "is_db", "epa_db", "is_nondb", "epa_nondb", "succ", "sk",
            "gv", "expl", "pr_obs", "pr"]
    off = plays.group_by(["season", "week", "posteam"]).agg([pl.col(c).sum() for c in cols]) \
               .rename({"posteam": "team"}).with_columns(pl.lit(1.0).alias("g"))
    dff = plays.group_by(["season", "week", "defteam"]).agg([pl.col(c).sum() for c in cols]) \
               .rename({"defteam": "team"}).with_columns(pl.lit(1.0).alias("g"))
    st = pbp.filter((pl.col("special_teams_play") == 1) & pl.col("epa").is_not_null()
                    & pl.col("posteam").is_not_null())
    st_pos = st.group_by(["season", "week", "posteam"]).agg(pl.col("epa").sum().alias("a")).rename({"posteam": "team"})
    st_def = st.group_by(["season", "week", "defteam"]).agg(pl.col("epa").sum().alias("b")).rename({"defteam": "team"})
    st_net = st_pos.join(st_def, on=["season", "week", "team"], how="full", coalesce=True) \
                   .with_columns((pl.col("a").fill_null(0) - pl.col("b").fill_null(0)).alias("st_epa"),
                                 pl.lit(1.0).alias("g")).select(["season", "week", "team", "st_epa", "g"])
    games = off.select(["season", "week", "team"])
    return off, dff, st_net, games


# ---------------------------------------------------------------- QB (produccion)
def qb_season_values(pbp: pl.DataFrame) -> pl.DataFrame:
    """qb_adjustment.qb_career_value vectorizado: valor acumulado hasta cada temporada."""
    db = pbp.filter((pl.col("qb_dropback") == 1) & pl.col("qb_epa").is_not_null()
                    & pl.col("passer_player_id").is_not_null())
    sq = db.group_by(["season", "passer_player_id"]).agg(pl.col("qb_epa").mean().alias("v"),
                                                         pl.len().alias("n")).sort(["passer_player_id", "season"])
    out = []
    for (qb,), g in sq.group_by(["passer_player_id"], maintain_order=True):
        ss = g["season"].to_numpy(); v = g["v"].to_numpy(); n = g["n"].to_numpy()
        sw = np.minimum(n / QB_MIN_DROPBACKS, 1.0)
        for i, s in enumerate(ss):
            w = QB_SEASON_DECAY ** (s - ss[:i + 1]) * sw[:i + 1]
            if w.sum() > 0:
                out.append((int(s), qb, float(np.average(v[:i + 1], weights=w))))
    return pl.DataFrame(out, schema={"season": pl.Int32, "qb_id": pl.Utf8, "qb_value": pl.Float64}, orient="row")


def bucket_expr(col="round"):
    return (pl.when(pl.col(col) == 1).then(pl.lit("R1"))
              .when(pl.col(col).is_in([2, 3])).then(pl.lit("R2_R3")).otherwise(pl.lit("R4_PLUS")))


def qb_value_for_games(tg: pl.DataFrame, qbv: pl.DataFrame, picks: pl.DataFrame) -> pl.DataFrame:
    """Valor del QB titular: ultimo valor con temporada < s; si no hay, cubeta de draft
    calculada SOLO con novatos de temporadas < s (correccion de leakage vs produccion)."""
    qpicks = picks.filter(pl.col("position") == "QB").select(
        pl.col("season").cast(pl.Int32).alias("draft_season"), "round", pl.col("gsis_id").alias("qb_id")
    ).drop_nulls("qb_id").unique("qb_id").with_columns(bucket_expr().alias("bucket"))
    rookie = qbv.join(qpicks, left_on=["season", "qb_id"], right_on=["draft_season", "qb_id"], how="inner")
    seasons = sorted(tg["season"].unique().to_list())
    base_rows = []
    glob = rookie["qb_value"].mean()
    for s in seasons:
        r = rookie.filter(pl.col("season") < s).group_by("bucket").agg(pl.col("qb_value").mean().alias("b"))
        d = dict(r.iter_rows())
        for b in ["R1", "R2_R3", "R4_PLUS"]:
            base_rows.append((s, b, d.get(b, glob)))
    base = pl.DataFrame(base_rows, schema={"season": pl.Int32, "bucket": pl.Utf8, "rookie_base": pl.Float64}, orient="row")

    hist = qbv.sort("season").rename({"season": "hist_season"})
    g = tg.select(["season", "week", "team", "qb_id"]).with_columns(pl.col("season").cast(pl.Int32))
    g = g.with_columns((pl.col("season") - 1).alias("_s1")).sort("_s1")
    g = g.join_asof(hist.with_columns(pl.col("hist_season").cast(pl.Int32)).sort("hist_season"),
                    left_on="_s1", right_on="hist_season", by="qb_id", strategy="backward")
    g = g.join(qpicks.select(["qb_id", "bucket"]), on="qb_id", how="left") \
         .with_columns(pl.col("bucket").fill_null("R4_PLUS")) \
         .join(base, on=["season", "bucket"], how="left") \
         .with_columns(pl.coalesce(["qb_value", "rookie_base"]).alias("team_qb_value"))
    return g.select(["season", "week", "team", "team_qb_value"])


def qb_live_value(tg: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """C4c: (150*prior + S)/(150 + N), S,N = suma decaida (0.85/semana) de qb_epa y
    dropbacks del QB titular en semanas previas de la temporada (cualquier equipo)."""
    db = pbp.filter((pl.col("qb_dropback") == 1) & pl.col("qb_epa").is_not_null()
                    & pl.col("passer_player_id").is_not_null())
    wk = db.group_by(["season", "week", "passer_player_id"]).agg(pl.col("qb_epa").sum().alias("S"),
                                                                 pl.len().cast(pl.Float64).alias("N"))
    j = tg.select(["season", "week", "team", "qb_id", "team_qb_value"]).join(
        wk.rename({"week": "w", "passer_player_id": "qb_id"}), on=["season", "qb_id"], how="left")
    j = j.with_columns(pl.when(pl.col("w") < pl.col("week"))
                       .then(DECAY ** (pl.col("week") - pl.col("w")).cast(pl.Float64)).otherwise(0.0).alias("dw"))
    agg = j.group_by(["season", "week", "team"]).agg(
        (pl.col("S").fill_null(0) * pl.col("dw")).sum().alias("S"),
        (pl.col("N").fill_null(0) * pl.col("dw")).sum().alias("N"),
        pl.col("team_qb_value").first())
    agg = agg.with_columns(((QB_LIVE_PRIOR_N * pl.col("team_qb_value") + pl.col("S")) /
                            (QB_LIVE_PRIOR_N + pl.col("N"))).alias("qb_value_live"))
    return agg.select(["season", "week", "team", "qb_value_live"])


def backup_qb_flag(tg: pl.DataFrame) -> pl.DataFrame:
    """C4d: 1 si el titular del partido no es el lider de titularidades previas del equipo
    en la temporada (empate -> el mas reciente). Sin partidos previos -> 0."""
    rows = []
    for (s, t), g in tg.sort("week").group_by(["season", "team"], maintain_order=True):
        starts, last = {}, {}
        for wk, qb in g.select(["week", "qb_id"]).iter_rows():
            if starts:
                lead = max(starts, key=lambda q: (starts[q], last[q]))
                flag = int(qb is not None and qb != lead)
            else:
                flag = 0
            rows.append((s, wk, t, flag))
            if qb is not None:
                starts[qb] = starts.get(qb, 0) + 1; last[qb] = wk
    return pl.DataFrame(rows, schema={"season": pl.Int32, "week": pl.Int32, "team": pl.Utf8, "backup_qb": pl.Int32}, orient="row")


# ---------------------------------------------------------------- disponibilidad
def availability(tg: pl.DataFrame, injuries: pl.DataFrame, snaps: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """C4a/C4b. Snap share de un jugador en la semana wk = promedio de su % de snaps en
    partidos REG previos de la temporada; si no tiene, promedio de la temporada previa."""
    idmap = players.select(["pfr_id", "gsis_id"]).drop_nulls().unique("pfr_id")
    sc = snaps.filter(pl.col("game_type") == "REG").join(idmap, left_on="pfr_player_id", right_on="pfr_id", how="left") \
              .select(["season", "week", "team", "pfr_player_id", "gsis_id", "position", "offense_pct", "defense_pct"]) \
              .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    inj = injuries.filter((pl.col("game_type") == "REG") & pl.col("report_status").is_in(["Out", "Doubtful"])) \
                  .sort("date_modified").unique(["season", "week", "team", "gsis_id"], keep="last") \
                  .select(["season", "week", "team", "gsis_id", "position"])
    seasons = sorted(x for x in set(tg["season"].to_list()) if x >= AVAIL_START)
    out = []
    for s in seasons:
        cur = sc.filter(pl.col("season") == s)
        prev = sc.filter(pl.col("season") == s - 1)
        prev_pl = prev.group_by("pfr_player_id").agg(pl.col("offense_pct").mean().alias("o"), pl.col("defense_pct").mean().alias("d"),
                                                      pl.col("gsis_id").first())
        prev_team = prev.group_by(["team", "pfr_player_id"]).agg(pl.col("offense_pct").mean().alias("o"),
                                                                 pl.col("defense_pct").mean().alias("d"),
                                                                 pl.col("position").last(), pl.col("gsis_id").first())
        weeks = sorted(tg.filter(pl.col("season") == s)["week"].unique().to_list())
        for wk in weeks:
            past = cur.filter(pl.col("week") < wk)
            pl_share = past.group_by("pfr_player_id").agg(pl.col("offense_pct").mean().alias("o"),
                                                          pl.col("defense_pct").mean().alias("d"), pl.col("gsis_id").first())
            # share por gsis: actual si existe, si no temporada previa
            share = pl.concat([pl_share, prev_pl.join(pl_share.select("pfr_player_id"), on="pfr_player_id", how="anti")]) \
                      .drop_nulls("gsis_id").unique("gsis_id")
            outs = inj.filter((pl.col("season") == s) & (pl.col("week") == wk)).join(
                share.select(["gsis_id", "o", "d"]), on="gsis_id", how="left").fill_null(0.0)
            ol = outs.filter(pl.col("position").is_in(list(OL_POS))).group_by("team").agg(pl.col("o").sum().alias("ol_out"))
            dl = outs.filter(pl.col("position").is_in(list(DL_POS))).group_by("team").agg(pl.col("d").sum().alias("dl_out"))
            # WR1 / CB1 del equipo: mayor share previo en la posicion (temporada actual, o previa si wk sin datos)
            team_tbl = past.group_by(["team", "pfr_player_id"]).agg(pl.col("offense_pct").mean().alias("o"),
                                                                    pl.col("defense_pct").mean().alias("d"),
                                                                    pl.col("position").last(), pl.col("gsis_id").first())
            if team_tbl.height == 0:
                team_tbl = prev_team
            wr1 = team_tbl.filter(pl.col("position") == "WR").sort("o", descending=True).unique("team", keep="first")
            cb1 = team_tbl.filter(pl.col("position") == "CB").sort("d", descending=True).unique("team", keep="first")
            out_ids = outs.select(["team", "gsis_id"]).with_columns(pl.lit(1).alias("is_out"))
            wr1 = wr1.join(out_ids, on=["team", "gsis_id"], how="left").with_columns(
                (pl.col("o") * pl.col("is_out").fill_null(0)).alias("wr1_out")).select(["team", "wr1_out"])
            cb1 = cb1.join(out_ids, on=["team", "gsis_id"], how="left").with_columns(
                (pl.col("d") * pl.col("is_out").fill_null(0)).alias("cb1_out")).select(["team", "cb1_out"])
            base = tg.filter((pl.col("season") == s) & (pl.col("week") == wk)).select(["season", "week", "team"])
            for t in (ol, dl, wr1, cb1):
                base = base.join(t, on="team", how="left")
            out.append(base.fill_null(0.0))
    if not out:
        return pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32, "team": pl.Utf8, "ol_out": pl.Float64,
                                    "dl_out": pl.Float64, "wr1_out": pl.Float64, "cb1_out": pl.Float64})
    return pl.concat(out, how="diagonal_relaxed").with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))


# ---------------------------------------------------------------- todo junto
def build_features(raw: dict, seasons=None) -> pl.DataFrame:
    tg_all = team_games(raw["schedules"]).with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    seasons = seasons or list(range(FIRST_SEASON, LAST_SEASON + 1))
    tg = tg_all.filter(pl.col("season").is_in(seasons))
    pbp = raw["pbp"].with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    off, dff, st, games = pbp_aggregates(pbp, raw["participation"])

    T = tg.select(["season", "week", "team"]).unique()
    def feat(agg, num, den, name, side="team"):
        f = decayed_blend(agg, num, den, games, T, name)
        if side == "opp":
            f = f.rename({"team": "opp"})
            return f
        return f

    specs = [  # (agg, num, den, nombre, lado)
        (off, "epa_run", "is_run", "team_rush_epa", "team"),
        (dff, "epa", "n", "opp_def_epa_allowed", "opp"),
        (off, "epa_db", "is_db", "team_pass_epa", "team"),
        (off, "succ", "n", "team_success", "team"),
        (dff, "epa_db", "is_db", "opp_def_pass_epa", "opp"),
        (dff, "epa_nondb", "is_nondb", "opp_def_rush_epa", "opp"),
        (off, "gv", "n", "team_giveaway_raw", "team"),
        (dff, "gv", "n", "opp_takeaway_raw", "opp"),
        (off, "sk", "is_db", "team_sack_allowed", "team"),
        (dff, "sk", "is_db", "opp_sack_rate", "opp"),
        (off, "pr", "pr_obs", "team_pressure_allowed", "team"),
        (dff, "pr", "pr_obs", "opp_pressure_rate", "opp"),
        (off, "expl", "n", "team_explosive", "team"),
        (dff, "expl", "n", "opp_explosive_allowed", "opp"),
        (st, "st_epa", "g", "team_st_epa", "team"),
        (st, "st_epa", "g", "opp_st_epa", "opp"),
        (off, "n", "g", "team_pace", "team"),
        (off, "n", "g", "opp_pace", "opp"),
    ]
    df = tg
    for agg, num, den, name, side in specs:
        f = feat(agg, num, den, name)
        key = "team" if side == "team" else "opp"
        df = df.join(f.rename({"team": key}), on=["season", "week", key], how="left")

    # entregas: encogidas 50% hacia la media de liga de la temporada previa
    lg_gv = league_prior(off, "gv", "n", seasons)
    df = df.join(lg_gv, on="season", how="left").with_columns(
        (pl.col("league") + TURNOVER_SHRINK * (pl.col("team_giveaway_raw") - pl.col("league"))).alias("team_giveaway"),
        (pl.col("league") + TURNOVER_SHRINK * (pl.col("opp_takeaway_raw") - pl.col("league"))).alias("opp_takeaway"),
    ).drop("league")

    # QB
    qbv = qb_season_values(pbp)
    qb = qb_value_for_games(tg, qbv, raw["draft_picks"])
    df = df.join(qb, on=["season", "week", "team"], how="left")
    df = df.join(qb_live_value(df, pbp), on=["season", "week", "team"], how="left")
    df = df.join(backup_qb_flag(tg), on=["season", "week", "team"], how="left")

    # disponibilidad (2013+)
    av = availability(tg, raw["injuries"], raw["snap_counts"], raw["players"])
    df = df.join(av.select(["season", "week", "team", "ol_out", "wr1_out"]), on=["season", "week", "team"], how="left")
    df = df.join(av.select(["season", "week", pl.col("team").alias("opp"), pl.col("dl_out").alias("opp_dl_out"),
                            pl.col("cb1_out").alias("opp_cb1_out")]), on=["season", "week", "opp"], how="left")

    # estadio / clima / mercado
    indoor = pl.col("roof").is_in(["dome", "closed"])
    df = df.with_columns([
        indoor.cast(pl.Float64).fill_null(0).alias("indoor"),
        (pl.col("surface").str.strip_chars().fill_null("") != "grass").cast(pl.Float64).alias("turf"),
        pl.when(indoor).then(0.0).otherwise(pl.col("wind").cast(pl.Float64)).alias("wind_obs"),
        pl.when(indoor).then(0.0).otherwise((50.0 - pl.col("temp").cast(pl.Float64)).clip(lower_bound=0)).alias("cold_obs"),
        (pl.col("total_line") / 2 + pl.when(pl.col("is_home") == 1).then(pl.col("spread_line") / 2)
         .otherwise(-pl.col("spread_line") / 2)).clip(lower_bound=3.0).log().alias("log_implied"),
    ])
    return df.sort(["season", "week", "game_id", "is_home"])


if __name__ == "__main__":
    import time
    t0 = time.time()
    raw = load_raw()
    df = build_features(raw)
    df.write_parquet(DATA / "features.parquet")
    print(df.shape, f"{time.time() - t0:.1f}s")
    print(df.filter(pl.col("season") == 2024).describe().transpose(include_header=True))
