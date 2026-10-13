"""
v4: vuelve a tag_slug='nfl' (ya confirmado que funciona), con el filtro
de titulo corregido (el nombre real es "Big Game Champion", no "Super Bowl
Champion" -- mi error en la version anterior).
"""
import requests
import json
import time
from datetime import datetime

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def search_champion_events():
    all_events = []
    offset = 0
    while True:
        try:
            r = requests.get(f"{GAMMA_BASE}/events", params={
                "tag_slug": "nfl", "closed": "true", "limit": 100, "offset": offset,
            }, timeout=15)
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            print(f"Error (offset={offset}): {e}")
            break
        if not batch:
            break
        all_events.extend(batch)
        offset += 100
        time.sleep(0.25)

    champion_events = [
        e for e in all_events
        if "champion" in e.get("title", "").lower()
        and ("big game" in e.get("title", "").lower() or "super bowl" in e.get("title", "").lower())
    ]
    return all_events, champion_events


def get_preseason_price(clob_token_id, event_start_ts):
    ref_ts = event_start_ts + 86400 * 3
    end_ts = ref_ts + 86400 * 7
    try:
        r = requests.get(f"{CLOB_BASE}/prices-history", params={
            "market": clob_token_id, "startTs": ref_ts, "endTs": end_ts, "fidelity": 1440
        }, timeout=15)
        r.raise_for_status()
        history = r.json().get("history", [])
        return history[0]["p"] if history else None
    except Exception:
        return None


if __name__ == "__main__":
    print("Buscando eventos NFL cerrados (tag_slug='nfl')...\n")
    all_events, champion_events = search_champion_events()
    print(f"Total eventos NFL cerrados encontrados: {len(all_events)}")
    print(f"De esos, eventos de 'campeon': {len(champion_events)}\n")

    for e in champion_events:
        print(f"  - {e.get('title')}  |  slug={e.get('slug')}  |  startDate={e.get('startDate')}")

    if not champion_events:
        print("\nNo hay mas que el que ya conocemos. Muestra de otros titulos NFL encontrados (primeros 15),")
        print("por si el nombre de temporadas viejas es distinto a lo esperado:")
        for e in all_events[:15]:
            print(f"  - {e.get('title')}")
        exit()

    events_sorted = sorted(champion_events, key=lambda e: e.get("startDate", ""))
    oldest_event = events_sorted[0]

    print(f"\n{'='*60}\nDetalle completo (evento mas antiguo): {oldest_event.get('title')}\n{'='*60}")
    start_dt = datetime.fromisoformat(oldest_event.get("startDate").replace("Z", "+00:00"))
    start_ts = int(start_dt.timestamp())

    markets = oldest_event.get("markets", [])
    results = []
    for m in markets:
        question = m.get("question", "")
        clob_ids_raw = m.get("clobTokenIds")
        try:
            clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
            yes_token = clob_ids[0]
        except Exception:
            continue
        price = get_preseason_price(yes_token, start_ts)
        results.append({"question": question, "preseason_prob": price})
        time.sleep(0.25)

    results = [r for r in results if r["preseason_prob"] is not None]
    results.sort(key=lambda x: x["preseason_prob"], reverse=True)

    print(f"\nTabla completa ({len(results)} equipos), ordenada por probabilidad de preseason:")
    for i, r in enumerate(results, 1):
        print(f"  {i:2d}. {r['question']:<45} {r['preseason_prob']*100:.1f}%")