"""
Spreads de Vegas 2026, pegados a mano cada semana (estilo starters_2026.py).

Convencion: MARKET_SPREADS[(visita, local)] = margen esperado del LOCAL.
  positivo = favorece al local, negativo = favorece a la visita.
  Ojo: Vegas publica la linea del lado de cada equipo. "ATL +6" (local perro de 6)
  se escribe -6.0; "DAL -9.5" (local favorito de 9.5) se escribe +9.5.
Equipos con el mismo codigo que nflreadpy (LA = Rams, LV, LAC, WAS...).
Partidos sin linea -> el pipeline usa solo el modelo (Elo + Poisson).
"""

MARKET_SPREADS = {
    # --- Semana 5 (lineas de mitad de semana) ---
    ("BAL", "ATL"): -6.0,
    ("BUF", "LA"): +2.5,
    ("CHI", "GB"): -2.5,
    ("CIN", "MIA"): -7.0,
    ("CLE", "NYJ"): +2.5,   # orientacion por confirmar
    ("DEN", "LAC"): -2.5,
    ("DET", "ARI"): -5.5,
    ("HOU", "TEN"): -6.0,
    ("IND", "PIT"): +2.5,
    ("LV", "NE"): +3.5,
    ("MIN", "NO"): -1.5,
    ("NYG", "WAS"): -3.0,
    ("PHI", "JAX"): +6.5,   # campo neutral (Londres); la linea ya lo incorpora
    ("SF", "SEA"): +3.0,
    ("TB", "DAL"): +9.5,
}
