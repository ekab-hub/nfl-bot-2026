"""
Arma el diccionario {team: gsis_id} para 2026 a partir de nombres (mas facil
de mantener que IDs a mano), cruzando contra el roster real de nflreadpy.
Equipos marcados como 'inciertos' quedan documentados como limitacion del
pronostico, no resueltos por el codigo.
"""
import nflreadpy as nfl
import polars as pl

# Titulares proyectados 2026, segun investigacion propia (julio 2026).
# Equipos con competencia de puesto abierta -> se uso el favorito indicado.
PROJECTED_STARTERS = {
    "ARI": "Jacoby Brissett",
    "ATL": "Michael Penix Jr.",       # incierto: compite con Michael Penix
    "BAL": "Lamar Jackson",
    "BUF": "Josh Allen",
    "CAR": "Bryce Young",
    "CHI": "Caleb Williams",
    "CIN": "Joe Burrow",
    "CLE": "Deshaun Watson",      # incierto: compite con Deshaun Watson
    "DAL": "Dak Prescott",
    "DEN": "Bo Nix",
    "DET": "Jared Goff",
    "GB": "Jordan Love",
    "HOU": "C.J. Stroud",
    "IND": "Daniel Jones",
    "JAX": "Trevor Lawrence",
    "KC": "Patrick Mahomes",
    "LV": "Kirk Cousins",          # incierto: Fernando Mendoza entra en algun punto
    "LAC": "Justin Herbert",
    "LA": "Matthew Stafford",
    "MIA": "Malik Willis",
    "MIN": "Kyler Murray",         # incierto: compite con JJ McCarthy
    "NE": "Drake Maye",
    "NO": "Tyler Shough",
    "NYG": "Jameis Winston",
    "NYJ": "Geno Smith",
    "PHI": "Jalen Hurts",
    "PIT": "Aaron Rodgers",
    "SF": "Brock Purdy",
    "SEA": "Drew Lock",
    "TB": "Jalon Daniels",
    "TEN": "Cam Ward",
    "WAS": "Jayden Daniels",
}

UNCERTAIN_TEAMS = {"ATL", "CLE", "LV", "MIN"}  # para documentar en el reporte, no para resolver aqui


def build_current_starters_2026():
    rosters = nfl.load_rosters(seasons=[2026])
    qbs = rosters.filter(pl.col("position") == "QB").select(["team", "full_name", "gsis_id"])

    current_starters = {}
    not_found = []

    for team, name in PROJECTED_STARTERS.items():
        match = qbs.filter((pl.col("team") == team) & (pl.col("full_name") == name))
        if match.height == 0:
            # intento flexible: solo por nombre, sin filtrar equipo (por si el roster
            # todavia no refleja una firma/trade reciente)
            match = qbs.filter(pl.col("full_name") == name)
        if match.height == 0:
            not_found.append((team, name))
            continue
        current_starters[team] = match["gsis_id"][0]

    if not_found:
        print("ADVERTENCIA: no se encontraron estos jugadores en el roster de nflreadpy:")
        for team, name in not_found:
            print(f"  {team}: '{name}' -- revisar ortografia exacta o si el roster aun no se actualiza")

    print(f"\nTitulares resueltos: {len(current_starters)}/32")
    return current_starters


if __name__ == "__main__":
    starters = build_current_starters_2026()
    print(starters)