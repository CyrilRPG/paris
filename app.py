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
# --- Paramètres généraux
# =========================
BASE_URL = "https://v3.football.api-sports.io"
TIMEOUT = 25

DEFAULT_API_KEY = "507f1667eee8f206724727a95bfb6f6a"  # ta clé (tu peux l'enlever si tu passes par st.secrets)
DEFAULT_PROVIDER = "API-SPORTS"  # ou "RapidAPI"

PLAYER_FIRSTNAME = "Barnabás"
PLAYER_LASTNAME = "Varga"

# =========================
# --- Petites utilitaires
# =========================
def norm_str(s: str) -> str:
    return unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode().lower().strip()

def api_headers(provider: str, api_key: str) -> Dict[str, str]:
    if provider == "API-SPORTS":
        return {"x-apisports-key": api_key}
    else:
        return {"x-rapidapi-key": api_key, "x-rapidapi-host": "v3.football.api-sports.io"}

def http_get(path: str, headers: Dict[str, str], params: Dict[str, Any] = None) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def yesno(b: bool) -> int:
    return 1 if b else 0

def pick_first(lst: List[Any], pred) -> Optional[Any]:
    for x in lst:
        if pred(x):
            return x
    return None

# =========================
# --- Résolution joueur
# =========================
def resolve_player_id(headers: Dict[str, str], firstname: str, lastname: str) -> int:
    data = http_get("players/profiles", headers, params={"search": lastname})
    cand = data.get("response", [])
    if not cand:
        raise RuntimeError(f"Aucun profil trouvé pour {lastname}")

    def is_match(p):
        pf = p.get("firstname") or ""
        pl = p.get("lastname") or p.get("name") or ""
        nat = (p.get("nationality") or "").lower()
        ok_first = norm_str(pf).startswith(norm_str(firstname)) or norm_str(p.get("name","")).startswith(norm_str(firstname))
        ok_last = norm_str(pl).endswith(norm_str(lastname))
        return ok_first and ok_last and ("hungary" in nat or "magyar" in nat or nat == "hungary")

    player = pick_first(cand, is_match) or pick_first(
        cand, lambda p: norm_str(p.get("name","")).startswith(norm_str(firstname)) and norm_str(p.get("name","")).endswith(norm_str(lastname))
    )
    if not player or not player.get("id"):
        raise RuntimeError("Impossible d’identifier de manière unique le joueur.")
    return int(player["id"])

def get_player_teams(headers: Dict[str, str], player_id: int) -> List[Dict[str, Any]]:
    data = http_get("players/teams", headers, params={"player": player_id})
    return data.get("response", [])

def infer_current_team_and_season(player_teams: List[Dict[str, Any]], override_season: Optional[int]) -> Tuple[int, int]:
    if not player_teams:
        raise RuntimeError("Aucune équipe trouvée pour ce joueur.")
    latest_year = -1
    team_id = None
    season_year = None
    for entry in player_teams:
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
# --- Récup données match
# =========================
def get_team_fixtures(headers: Dict[str, str], team_id: int, season: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures", headers, params={"team": team_id, "season": season})
    return data.get("response", [])

def get_fixture_players(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/players", headers, params={"fixture": fixture_id})
    return data.get("response", [])

def get_fixture_events(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/events", headers, params={"fixture": fixture_id})
    return data.get("response", [])

def get_fixture_lineups(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/lineups", headers, params={"fixture": fixture_id})
    return data.get("response", [])

def get_fixture_odds(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    try:
        data = http_get("odds", headers, params={"fixture": fixture_id})
        return data.get("response", [])
    except Exception:
        # Certaines ligues n’ont pas d’odds
        return []

def extract_player_stats_from_fixture(fixture_players_payload: List[Dict[str, Any]], player_id: int) -> Dict[str, Any]:
    # cherche la fiche stats du joueur
    for teamblock in fixture_players_payload:
        for p in teamblock.get("players", []):
            try:
                if int(p.get("player", {}).get("id")) == player_id:
                    stats = (p.get("statistics") or [{}])[0]
                    return {
                        "minutes": stats.get("games", {}).get("minutes", 0) or 0,
                        "goals": stats.get("goals", {}).get("total", 0) or 0,
                    }
            except Exception:
                continue
    return {"minutes": 0, "goals": 0}

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
    # opponent_side in {"Home","Away"}
    for o in odds_payload:
        for book in o.get("bookmakers", []):
            for bet in book.get("bets", []):
                name = (bet.get("name") or "").lower()
                if name in {"match winner", "1x2", "winner", "match result"}:
                    for v in bet.get("values", []):
                        val = (v.get("value") or "").lower()
                        if opponent_side == "home" and val == "home":
                            try:
                                return float(v.get("odd"))
                            except Exception:
                                pass
                        if opponent_side == "away" and val == "away":
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
# --- xG via Understat (optionnel)
# =========================
def compute_match_xg_understat(understat_player_id: Optional[int], match_date_iso: str) -> Optional[float]:
    if not understat_player_id:
        return None
    try:
        import asyncio
        import nest_asyncio  # pour Streamlit/Jupyter
        nest_asyncio.apply()
        async def _run():
            import aiohttp
            from understat import Understat
            async with aiohttp.ClientSession() as session:
                u = Understat(session)
                shots = await u.get_player_shots(int(understat_player_id), {"season": None})
                xg = 0.0
                for s in shots:
                    if str(s.get("date",""))[:10] == match_date_iso:
                        try:
                            xg += float(s.get("xG", 0.0))
                        except Exception:
                            pass
                return xg
        return asyncio.get_event_loop().run_until_complete(_run())
    except Exception:
        return None

# =========================
# --- UI Streamlit
# =========================
st.set_page_config(page_title="Barnabás Varga — Match Extractor", layout="wide")
st.title("Barnabás Varga — Extracteur de matchs (API-FOOTBALL)")

with st.sidebar:
    st.header("🔧 Paramètres")
    provider = st.selectbox("Fournisseur", ["API-SPORTS", "RapidAPI"], index=0 if DEFAULT_PROVIDER=="API-SPORTS" else 1)
    # Priorité au secret si dispo
    secret_key = st.secrets.get("API_KEY") if hasattr(st, "secrets") else None
    api_key = st.text_input("API Key", value=secret_key or DEFAULT_API_KEY, type="password")
    season = st.text_input("Saison (YYYY) — optionnel", value="")
    us_id = st.text_input("Understat Player ID (optionnel)", value="", help="Permet de calculer l’xG par match")
    run_btn = st.button("▶️ Lancer l’extraction")

st.caption("Astuce : pour la prod publique, définis `API_KEY` dans **Secrets** (Streamlit Cloud) — voir les instructions plus bas.")

placeholder = st.empty()

def export_csv_download(df: pd.DataFrame, filename: str = "varga_matches.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger CSV", data=csv, file_name=filename, mime="text/csv")

if run_btn:
    try:
        if not api_key.strip():
            st.error("Merci de renseigner la clé API.")
            st.stop()

        headers = api_headers(provider, api_key.strip())

        with st.spinner("Vérification du statut d’abonnement…"):
            try:
                status = http_get("status", headers)
                sub = status.get("response", {}).get("subscription", {})
                st.success(f"Plan: {sub.get('plan')} | actif: {sub.get('active')}")
            except Exception as e:
                st.warning(f"Impossible de lire /status (non bloquant) : {e}")

        with st.spinner("Résolution de l’ID joueur…"):
            pid = resolve_player_id(headers, PLAYER_FIRSTNAME, PLAYER_LASTNAME)
            st.write(f"**Joueur**: {PLAYER_FIRSTNAME} {PLAYER_LASTNAME} — id = `{pid}`")

        with st.spinner("Récupération équipes & saison…"):
            teams = get_player_teams(headers, pid)
            override_season = int(season) if season.strip().isdigit() else None
            team_id, season_year = infer_current_team_and_season(teams, override_season)
            st.write(f"**Équipe courante** id = `{team_id}`, **saison** = `{season_year}`")

        with st.spinner("Récupération des fixtures de l’équipe…"):
            fixtures = get_team_fixtures(headers, team_id, season_year)
            st.write(f"{len(fixtures)} fixtures trouvées.")

        top_assists_cache: Dict[Tuple[int,int], List[int]] = {}
        rows = []

        prog = st.progress(0.0)
        total = max(1, len(fixtures))

        for idx, fx in enumerate(fixtures, start=1):
            fixture = fx.get("fixture", {})
            league = fx.get("league", {})
            teams_obj = fx.get("teams", {})

            fixture_id = fixture.get("id")
            date_iso = (fixture.get("date") or "")[:10]
            league_id = league.get("id")
            round_str = league.get("round") or ""
            home_team_id = teams_obj.get("home", {}).get("id")
            away_team_id = teams_obj.get("away", {}).get("id")
            home_name = teams_obj.get("home", {}).get("name")
            away_name = teams_obj.get("away", {}).get("name")

            if not fixture_id or not home_team_id or not away_team_id:
                prog.progress(idx/total)
                continue

            if team_id == home_team_id:
                dom_ext = "D"
                opponent_team_id = away_team_id
                opponent_name = away_name
                opponent_side = "away"
            else:
                dom_ext = "E"
                opponent_team_id = home_team_id
                opponent_name = home_name
                opponent_side = "home"

            # ---- stats joueur (minutes/goals)
            fp = get_fixture_players(headers, fixture_id)
            pstats = extract_player_stats_from_fixture(fp, pid)
            minutes = int(pstats.get("minutes", 0))
            goals = int(pstats.get("goals", 0))
            scored = goals > 0

            # ---- events -> penalty tiré ?
            evs = get_fixture_events(headers, fixture_id)
            took_pen = player_took_penalty(evs, pid)

            # ---- lineups
            lineups = get_fixture_lineups(headers, fixture_id)

            # ---- odds pré-match -> cote de l’adversaire
            odds_payload = get_fixture_odds(headers, fixture_id)
            opp_price = extract_opponent_price(odds_payload, opponent_side)

            # ---- top assists présent ?
            key_assister_present = 0
            if league_id:
                key = (int(league_id), int(season_year))
                if key not in top_assists_cache:
                    top_assists_cache[key] = get_top_assisters_ids(headers, league_id, season_year)
                key_assister_present = yesno(any_of_players_in_lineups(lineups, top_assists_cache[key]))

            # ---- importance binaire
            important = yesno(is_important_round(round_str))

            # ---- xG (Understat) optionnel
            xg = None
            if us_id.strip():
                try:
                    xg_val = compute_match_xg_understat(int(us_id.strip()), date_iso)
                    if xg_val is not None and not (isinstance(xg_val, float) and (math.isnan(xg_val) or math.isinf(xg_val))):
                        xg = round(float(xg_val), 3)
                except Exception:
                    xg = None

            rows.append({
                "date": date_iso,
                "adversaire": opponent_name,
                "dom_ext": dom_ext,
                "minutes": minutes,
                "but": int(scored),
                "tire_pen": int(took_pen),
                "xG": xg,
                "cote_adversaire_1x2": opp_price,
                "passeur_principal_present": key_assister_present,
                "important": important,
                "fixture_id": fixture_id,
                "league_id": league_id,
                "team_id": team_id,
                "opponent_team_id": opponent_team_id,
                "round": round_str,
            })

            # Petit délai pour limiter le RPS
            time.sleep(0.12)
            prog.progress(min(1.0, idx/total))

        df = pd.DataFrame(rows).sort_values(["date", "fixture_id"])
        st.subheader("Résultat")
        st.dataframe(df, use_container_width=True)
        export_csv_download(df)

    except Exception as e:
        st.error(f"Erreur : {e}")
