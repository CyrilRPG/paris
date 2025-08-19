import os
import time
import math
import io
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# =========================
# --- Config / constantes
# =========================
BASE_URL = "https://v3.football.api-sports.io"
TIMEOUT = 25

def norm(s: str) -> str:
    return unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode().lower().strip()

def api_headers(provider: str, api_key: str) -> Dict[str, str]:
    if provider == "API-SPORTS":
        return {"x-apisports-key": api_key}
    # RapidAPI
    return {"x-rapidapi-key": api_key, "x-rapidapi-host": "v3.football.api-sports.io"}

def http_get(path: str, headers: Dict[str, str], params: Dict[str, Any] = None) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def yesno(b: bool) -> int:
    return 1 if b else 0

def pick_first(lst, pred):
    for x in lst:
        if pred(x):
            return x
    return None

# =========================
# --- Recherche équipe / joueur
# =========================
def search_team_id(headers: Dict[str, str], team_query: str) -> Optional[int]:
    if not team_query.strip():
        return None
    data = http_get("teams", headers, params={"search": team_query})
    for r in data.get("response", []):
        tid = r.get("team", {}).get("id")
        if tid:
            return int(tid)
    return None

def list_squad_candidates(headers: Dict[str, str], team_id: int, player_query: str) -> List[Dict[str, Any]]:
    resp = http_get("players/squads", headers, params={"team": team_id}).get("response", [])
    q = norm(player_query)
    cands = []
    for blk in resp:
        for p in blk.get("players", []):
            nm = p.get("name") or ""
            if all(tok in norm(nm) for tok in q.split()):
                cands.append({"id": p.get("id"), "name": p.get("name"), "age": p.get("age"), "position": p.get("position")})
    return cands

def profiles_by_lastname(headers: Dict[str, str], player_query: str) -> List[Dict[str, Any]]:
    # On tente d'abord la "last name" (ou partie du nom) via /players/profiles
    last = player_query.split()[-1]
    prof = http_get("players/profiles", headers, params={"search": last}).get("response", [])
    # Scoring simple par similarité grossière
    q = norm(player_query)
    scored = []
    for p in prof:
        cand = (p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip())
        score = sum(tok in norm(cand) for tok in q.split())
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored]

def player_current_team_and_season(headers: Dict[str, str], player_id: int, override_season: Optional[int]) -> Tuple[int, int]:
    data = http_get("players/teams", headers, params={"player": player_id}).get("response", [])
    if not data:
        raise RuntimeError("Aucune équipe trouvée pour ce joueur.")
    latest_year = -1
    team_id = None
    season_year = None
    for entry in data:
        tid = entry.get("team", {}).get("id")
        for s in entry.get("seasons", []):
            y = s.get("year")
            if isinstance(y, int) and y > latest_year:
                latest_year = y
                team_id = tid
                season_year = y
    if override_season:
        season_year = int(override_season)
    if not (team_id and season_year):
        raise RuntimeError("Impossible d’inférer équipe/saison.")
    return int(team_id), int(season_year)

# =========================
# --- Données par fixture
# =========================
def get_team_fixtures(headers: Dict[str, str], team_id: int, season: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures", headers, params={"team": team_id, "season": season})
    return data.get("response", [])

def get_fixture_players(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    return http_get("fixtures/players", headers, params={"fixture": fixture_id}).get("response", [])

def get_fixture_events(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    return http_get("fixtures/events", headers, params={"fixture": fixture_id}).get("response", [])

def get_fixture_lineups(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    return http_get("fixtures/lineups", headers, params={"fixture": fixture_id}).get("response", [])

def get_fixture_odds(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    try:
        return http_get("odds", headers, params={"fixture": fixture_id}).get("response", [])
    except Exception:
        return []

def extract_player_minutes_goals(fixture_players_payload: List[Dict[str, Any]], player_id: int) -> Tuple[int, int]:
    for teamblock in fixture_players_payload:
        for p in teamblock.get("players", []):
            try:
                if int(p.get("player", {}).get("id")) == player_id:
                    stats = (p.get("statistics") or [{}])[0]
                    minutes = stats.get("games", {}).get("minutes", 0) or 0
                    goals = stats.get("goals", {}).get("total", 0) or 0
                    return int(minutes), int(goals)
            except Exception:
                continue
    return 0, 0

def player_took_penalty(events: List[Dict[str, Any]], player_id: int) -> bool:
    for e in events:
        etype = (e.get("type") or "").lower()
        detail = (e.get("detail") or "").lower()
        pid = e.get("player", {}) and e.get("player", {}).get("id")
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid != player_id:
            continue
        if etype == "goal" and "penalty" in detail:
            return True
        if etype == "penalty":
            return True
    return False

def extract_opponent_price(odds_payload: List[Dict[str, Any]], opponent_side: str) -> Optional[float]:
    opp = opponent_side.lower()  # "home" / "away"
    for o in odds_payload:
        for book in o.get("bookmakers", []):
            for bet in book.get("bets", []):
                name = (bet.get("name") or "").lower()
                if name in {"match winner", "1x2", "winner", "match result"}:
                    for v in bet.get("values", []):
                        if (v.get("value") or "").lower() == opp:
                            try:
                                return float(v.get("odd"))
                            except Exception:
                                pass
    return None

def get_top_assisters_ids(headers: Dict[str, str], league_id: int, season: int) -> List[int]:
    try:
        data = http_get("players/topassists", headers, params={"league": league_id, "season": season})
        ids = []
        for r in data.get("response", []):
            pid = r.get("player", {}).get("id")
            if pid:
                ids.append(int(pid))
        return ids
    except Exception:
        return []

def any_of_players_in_lineups(lineups: List[Dict[str, Any]], player_ids: List[int]) -> bool:
    present = set()
    for team in lineups:
        for blk in ("startXI", "substitutes"):
            for p in team.get(blk, []):
                pid = p.get("player", {}).get("id")
                if pid:
                    present.add(int(pid))
    return len(set(player_ids) & present) > 0

def is_important_round(round_str: str) -> bool:
    s = (round_str or "").lower()
    keys = ["quarter", "semi", "final", "knockout", "play-offs", "playoffs", "round of", "barrage"]
    return any(k in s for k in keys)

# =========================
# --- UI Streamlit
# =========================
st.set_page_config(page_title="Player Match Extractor — API-FOOTBALL", layout="wide")
st.title("⚽ Player Match Extractor (API-FOOTBALL) — sans xG")

with st.sidebar:
    st.header("🔧 Paramètres")
    provider = st.selectbox("Fournisseur", ["API-SPORTS", "RapidAPI"], index=0)
    # Secrets > API_KEY (recommandé). Sinon champ texte.
    secret_key = st.secrets.get("API_KEY") if hasattr(st, "secrets") else None
    api_key = st.text_input("API Key", value=secret_key or "", type="password",
                            help="De préférence via Secrets (Streamlit Cloud).")
    season_in = st.text_input("Saison (YYYY) — optionnel", value="")
    team_hint = st.text_input("Équipe (optionnel)", value="", help="Aide à désambigüiser la recherche joueur.")
    player_query = st.text_input("🔎 Joueur (ex: Barnabás Varga)", value="")
    run_btn = st.button("▶️ Rechercher & extraire")

placeholder = st.empty()

def export_csv(df: pd.DataFrame, filename: str):
    st.download_button("📥 Télécharger CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")

if run_btn:
    try:
        if not api_key.strip():
            st.error("Merci de renseigner la clé API.")
            st.stop()
        if not player_query.strip():
            st.error("Merci de saisir un nom de joueur.")
            st.stop()

        headers = api_headers(provider, api_key.strip())

        # 0) statut (non bloquant)
        try:
            status = http_get("status", headers)
            sub = status.get("response", {}).get("subscription", {})
            st.success(f"Plan: {sub.get('plan')} | actif: {sub.get('active')}")
        except Exception as e:
            st.info(f"Info: impossible de lire /status (non bloquant).")

        # 1) Chercher par équipe si fournie
        candidates = []
        chosen_player_id: Optional[int] = None

        if team_hint.strip():
            tid = search_team_id(headers, team_hint)
            if tid:
                candidates = list_squad_candidates(headers, tid, player_query)

                if len(candidates) == 1:
                    chosen_player_id = int(candidates[0]["id"])
                elif len(candidates) > 1:
                    st.subheader("Plusieurs joueurs trouvés :")
                    label_map = {f"{c['name']} — {c.get('position','?')} — id:{c['id']}": c["id"] for c in candidates}
                    choice = st.selectbox("Sélectionne le joueur exact", list(label_map.keys()))
                    chosen_player_id = int(label_map[choice])
                else:
                    st.warning("Aucun joueur correspondant dans l’effectif indiqué. On tente une recherche globale…")

        # 2) Sinon / fallback : /players/profiles + sélection
        if not chosen_player_id:
            profs = profiles_by_lastname(headers, player_query)
            if not profs:
                raise RuntimeError("Aucun profil joueur trouvé.")
            # Construire des options lisibles (nom, nationalité, naissance)
            options = []
            for p in profs[:50]:
                pid = p.get("id")
                nm = p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip()
                nat = p.get("nationality") or "?"
                by  = p.get("birth", {}).get("date") or "?"
                options.append((f"{nm} — {nat} — id:{pid}", pid))
            label = st.selectbox("Sélectionne le joueur", [o[0] for o in options])
            chosen_player_id = int(dict(options)[label])

        st.write(f"**Joueur choisi** → id : `{chosen_player_id}`")

        # 3) Déterminer équipe & saison
        override_season = int(season_in) if season_in.strip().isdigit() else None
        team_id, season = player_current_team_and_season(headers, chosen_player_id, override_season)
        st.write(f"**Équipe** id `{team_id}` — **Saison** `{season}`")

        # 4) Fixtures de l’équipe pour la saison
        fixtures = get_team_fixtures(headers, team_id, season)
        st.write(f"{len(fixtures)} fixtures trouvées.")
        prog = st.progress(0.0)

        rows = []
        top_assists_cache: Dict[Tuple[int,int], List[int]] = {}
        total = max(1, len(fixtures))

        for idx, fx in enumerate(fixtures, start=1):
            fixture = fx.get("fixture", {})
            league  = fx.get("league", {})
            tms     = fx.get("teams", {})

            fixture_id = fixture.get("id")
            date_iso   = (fixture.get("date") or "")[:10]
            league_id  = league.get("id")
            round_str  = league.get("round") or ""
            home_id    = tms.get("home", {}).get("id")
            away_id    = tms.get("away", {}).get("id")
            home_name  = tms.get("home", {}).get("name")
            away_name  = tms.get("away", {}).get("name")

            if not fixture_id or not home_id or not away_id:
                prog.progress(idx/total); continue

            if team_id == home_id:
                dom_ext = "D"
                opponent_name = away_name
                opponent_side = "away"
            else:
                dom_ext = "E"
                opponent_name = home_name
                opponent_side = "home"

            # stats joueur
            fps = get_fixture_players(headers, fixture_id)
            minutes, goals = extract_player_minutes_goals(fps, chosen_player_id)
            scored = goals > 0

            # penalty ?
            evs = get_fixture_events(headers, fixture_id)
            took_pen = player_took_penalty(evs, chosen_player_id)

            # lineups
            lineups = get_fixture_lineups(headers, fixture_id)

            # odds (cote de l'adversaire en 1X2)
            odds = get_fixture_odds(headers, fixture_id)
            opp_price = extract_opponent_price(odds, opponent_side)

            # “passeur principal” présent ? (top assists ligue)
            key_assister_present = 0
            if league_id:
                key = (int(league_id), int(season))
                if key not in top_assists_cache:
                    top_assists_cache[key] = get_top_assisters_ids(headers, league_id, season)
                key_assister_present = yesno(any_of_players_in_lineups(lineups, top_assists_cache[key]))

            # importance binaire
            important = yesno(is_important_round(round_str))

            rows.append({
                "date": date_iso,
                "adversaire": opponent_name,
                "dom_ext": dom_ext,
                "minutes": minutes,
                "but": int(scored),
                "tire_pen": int(took_pen),
                "cote_adversaire_1x2": opp_price,
                "passeur_principal_present": key_assister_present,
                "important": important,
                "fixture_id": fixture_id,
                "league_id": league_id,
                "team_id": team_id,
                "round": round_str,
            })

            time.sleep(0.12)
            prog.progress(min(1.0, idx/total))

        df = pd.DataFrame(rows).sort_values(["date", "fixture_id"])
        st.subheader("Résultats")
        st.dataframe(df, use_container_width=True)
        export_csv(df, filename=f"player_{chosen_player_id}_{season}.csv")

    except Exception as e:
        st.error(f"Erreur : {e}")
