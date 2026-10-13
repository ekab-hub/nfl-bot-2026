import polars as pl
import weekly_pipeline as wp

SEASON, WEEK = 2026, 5
GAMES = {
    ("BAL", "ATL"): -6.0,
    ("CIN", "MIA"): -7.0,
    ("TB", "DAL"): 9.5,
    ("LV", "NE"): 3.5,
    ("NYG", "WAS"): -3.0,
    ("PHI", "JAX"): 6.5,
}
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(250)

elo = wp.get_current_elo_ratings(SEASON)
model, _ = wp.train_poisson_model(
    train_seasons=list(range(wp.POISSON_TRAIN_START, SEASON)),
    pbp_history_start=wp.POISSON_TRAIN_START - 1)
feats = wp.build_week_features(SEASON, WEEK)
w = wp.POISSON_BLEND_WEIGHT

rows = []
for (away, home), mkt in GAMES.items():
    hf = feats.filter((pl.col("home_team") == home) & (pl.col("side") == "home")).row(0, named=True)
    af = feats.filter((pl.col("away_team") == away) & (pl.col("side") == "away")).row(0, named=True)
    eh, ea = elo.get(home, 1500) + 35, elo.get(away, 1500)
    p_elo = 1 / (1 + 10 ** ((ea - eh) / 400))
    lh, la = wp.predict_lambda(model, hf), wp.predict_lambda(model, af)
    p_poi = wp.game_win_probs(lh, la)["home_win"]
    m_elo, m_poi = (eh - ea) / 25, lh - la
    rows.append({
        "partido": f"{away} @ {home}",
        "elo_local": round(eh - 35), "elo_visita": round(ea),
        "p_elo": round(p_elo, 3), "lam_local": round(lh, 1), "lam_visita": round(la, 1),
        "p_poisson": round(p_poi, 3), "p_blend": round(w * p_poi + (1 - w) * p_elo, 3),
        "m_elo": round(m_elo, 1), "m_poisson": round(m_poi, 1),
        "m_blend": round(w * m_poi + (1 - w) * m_elo, 1), "m_mercado": mkt,
    })
print(pl.DataFrame(rows))