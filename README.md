# NFL Bot 2026

Modelo estadístico y transparente para pronosticar la NFL, construido en un laboratorio universitario de analítica deportiva. Tiene dos partes:

1. **Pronóstico semanal** (`weekly_pipeline.py`): probabilidad de victoria y margen esperado de cada partido, mezclando Elo (55%) y una regresión de Poisson sobre puntos (45%).
2. **Simulación de temporada** (Monte Carlo, 10,000 simulaciones): probabilidad de ganar la división, la conferencia y el Super Bowl.

## Cómo funciona

**Elo.** Localía de 35 puntos, K=20, regresión a la media de 0.40 entre temporadas y multiplicador por margen de victoria. En la simulación condicionada a resultados reales se usa K=35 dentro de la temporada.

**Poisson (puntos por equipo).** Variables estandarizadas (alpha 1e-4), entrenado con datos desde 2002: EPA de carrera y de pase del ataque, success rate, valor del QB actualizado con la temporada, marca de QB suplente, EPA defensivo permitido por el rival, localía y partido divisional.

**Simulación de temporada.** Margen esperado = (diferencia de Elo + localía) / 25, con ruido normal de desviación 13.5. El Elo se actualiza tras cada partido simulado. Desempates simplificados y llave de 7 equipos por conferencia. Con `use_played=True` se condiciona a los resultados ya jugados.

**QB perdido por toda la temporada.** Ajuste manual (`qb_out_2026.py`), calibrado con las líneas de Vegas 2010-2025: unos 8.4 puntos de spread por unidad de valor de QB (un QB de élite contra un suplente típico son unos 2 puntos de spread, unos 52 puntos de Elo).

## Qué sabemos y qué no

- **Pronóstico semanal, holdout 2023-2025:** log-loss del blend 0.6250 (versión anterior 0.6340; Elo solo 0.6348). Accuracy 0.654.
- **Contra el mercado:** el modelo cierra cerca de la mitad de la brecha de log-loss con la línea de Vegas, y ninguno de los factores probados mejora a la línea. Por eso Vegas se muestra solo como referencia (`MARKET_BLEND_WEIGHT = 0.0`) y no se mezcla.
- **Simulación condicionada a resultados:** la proyección mejora con las semanas. Acierto por conferencia de los equipos que llegan a playoffs: top 8 de 0.482 (semana 0) a 0.589 (semana 4) y 0.643 (semana 8); top 4 de 0.274 a 0.393 y 0.476; top 2 de 0.214 a 0.357 y 0.429.
- **Rechazado con pruebas:** el ajuste por "suerte" (récord pitagórico) frente al Elo, y otras variables (lesiones, momentum, contratos, entre otras). Ver los scripts `*_significance_test.py` y `calibrar_luck.py`.
- Los protocolos y reportes completos del Poisson, el mercado y el QB están en `experimentos/`.

**Limitaciones.** Depende de entradas manuales (QB titulares, partidos en campo neutral, bajas de QB). No ve lesiones de última hora ni de otras posiciones. El holdout 2023-2025 no es virgen para los parámetros globales. No medimos cómo rinde el modelo semanal en las primeras semanas de temporada por separado.

## Instalación

```
git clone https://github.com/ekab-hub/nfl-bot-2026.git
cd nfl-bot-2026
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Probado con Python 3.13. La primera corrida descarga datos de nflverse (necesita internet) y tarda unos minutos.

## Uso semanal

Lo ideal es hacerlo el jueves o después, cuando ya hay reportes de lesiones.

1. Actualiza los QB titulares en `starters_2026.py` (`PROJECTED_STARTERS`). Los nombres deben coincidir con el roster.
2. Si hay un partido en campo neutral, agrégalo a `NEUTRAL_SITE_GAMES` en `weekly_pipeline.py` con el formato `(visita, local)`. Si no hay, déjalo vacío.
3. Opcional: pega las líneas de Vegas en `market_lines_2026.py` (spread del local, positivo si el local es favorito). Solo llenan una columna de comparación.
4. Corre `python3 weekly_pipeline.py`. Lee los avisos del inicio.
5. Para proyecciones de temporada: `python3 division_champions.py`, `python3 conference_champions.py` o `python3 superbowl_champions.py`. Si un QB titular se pierde toda la temporada, anótalo en `QB_OUT_FOR_SEASON` (`qb_out_2026.py`).

## Estructura

- **Producción:** `weekly_pipeline.py`, `motor_elo_v1.py`, `poisson_model.py`, `poisson_features.py`, `team_epa.py`, `qb_adjustment.py`.
- **Simulación de temporada:** `season_sim_core.py`, `division_champions.py`, `conference_champions.py`, `superbowl_champions.py`.
- **Entradas manuales:** `starters_2026.py`, `market_lines_2026.py`, `qb_out_2026.py`.
- **Validación y calibración:** `backtest_elo_v1.py`, `walk_forward_backtest.py`, `top_n_backtest.py`, `validar_condicionada.py`, `calibracion_en_temporada.py`, `calibrar_luck.py`, entre otros.
- **Variables evaluadas y descartadas:** `*_significance_test.py`.
- **`experimentos/`:** pruebas con protocolo preregistrado (factores del Poisson, mercado, QB). Los datos descargados no se incluyen; se regeneran con `experimentos/download_data.py`.

## Datos y créditos

Datos de [nflverse](https://github.com/nflverse) vía `nflreadpy`; consulta su licencia y condiciones de uso. Las líneas de mercado se capturan a mano y se usan solo como referencia.
