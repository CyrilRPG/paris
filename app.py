import time
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# =========================
# --- Config
# =========================
BASE_URL = "https://v3.football.api-sports.io"
TIMEOUT = 25

def norm(s: str) -> str:
    return unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode().lower().strip()

def api_headers(provider: str, api_key: str) -> Dict[str, str]:
    if provider == "API-SPORTS":
        return {"x-apisports-key": api_key}
    return {"x-rapidapi-key": api_key, "x-rapidapi-host": "v3.football.api-sports.io"}

def http_get(path: str, headers: Dict[str, str], params: Dict[str, Any] = None) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Réponse inattendue sur {path}: type={type(data)}")
    return data

def yesno(b: bool) -> int:
    return 1 if b else 0

# =========================
# --- Recherche équipe / joueur
# =========================
def search_team_id(headers: Dict[str, str], team_query: str) -> Optional[int]:
    if not team_query.strip():
        return None
    data = http_get("teams", headers, params={"search": team_query})
    for r in data.get("response", []) or []:
        if isinstance(r, dict):
            tid = (r.get("team") or {}).get("id")
            if isinstance(tid, int):
                return tid
    return None

def list_squad_candidates(headers: Dict[str, str], team_id: int, player_query: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    data = http_get("players/squads", headers, params={"team": team_id})
    q_tokens = [t for t in norm(player_query).split() if t]
    for blk in data.get("response", []) or []:
        if not isinstance(blk, dict):
            continue
        for p in blk.get("players", []) or []:
            if not isinstance(p, dict):
                continue
            nm = p.get("name") or ""
            nmn = norm(nm)
            if all(tok in nmn for tok in q_tokens):
                pid = p.get("id")
                if isinstance(pid, int):
                    out.append({
                        "id": pid,
                        "name": nm,
                        "age": p.get("age"),
                        "position": p.get("position")
                    })
    return out

def _normalize_profiles_list(resp: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for x in resp:
        if isinstance(x, dict):
            if "id" in x:
                out.append(x)
                continue
            if "player" in x and isinstance(x["player"], dict) and "id" in x["player"]:
                merged = dict(x["player"])
                if "name" not in merged and "name" in x:
                    merged["name"] = x["name"]
                out.append(merged)
    return out

def profiles_by_lastname(headers: Dict[str, str], player_query: str) -> List[Dict[str, Any]]:
    last = player_query.split()[-1]
    data = http_get("players/profiles", headers, params={"search": last})
    prof = _normalize_profiles_list(data.get("response", []) or [])
    qtokens = [t for t in norm(player_query).split() if t]
    scored = []
    for p in prof:
        cand = p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip()
        cn = norm(cand)
        score = sum(tok in cn for tok in qtokens)
        if score > 0 and isinstance(p.get("id"), int):
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored]

# =========================
# --- Trouver saison & équipe via /players
# =========================
def get_player_seasons(headers: Dict[str, str], player_id: int) -> List[int]:
    data = http_get("players/seasons", headers, params={"player": player_id})
    seasons = data.get("response", []) or []
    return [int(s) for s in seasons if isinstance(s, int)]

def player_team_from_players_endpoint(headers: Dict[str, str], player_id: int, season: int) -> Tuple[int, Dict[str, Any]]:
    """
    Utilise /players?id=PID&season=YYYY pour déterminer l'équipe.
    Retourne (team_id, payload brut du bloc 'statistics' choisi).
    Choix : on prend l'entrée avec le PLUS DE MINUTES (si transfert).
    """
    data = http_get("players", headers, params={"id": player_id, "season": season})
    resp = data.get("response", []) or []
    if not resp:
        raise RuntimeError("Aucune statistique trouvée pour ce joueur et cette saison.")
    # agrège tous les blocs statistics[*]
    stats_blocks = []
    for rec in resp:
        if not isinstance(rec, dict):
            continue
        for st in rec.get("statistics", []) or []:
            if isinstance(st, dict):
                team = st.get("team") or {}
                tid = team.get("id")
                if isinstance(tid, int):
                    minutes = 0
                    games = st.get("games") or {}
                    minutes = int(games.get("minutes", 0) or 0) if isinstance(games, dict) else 0
                    stats_blocks.append((minutes, tid, st))
    if not stats_blocks:
        raise RuntimeError("Impossible d’extraire l’équipe depuis /players.")
    stats_blocks.sort(key=lambda t: -t[0])  # plus de minutes en premier
    return stats_blocks[0][1], stats_blocks[0][2]

def resolve_team_and_season(headers: Dict[str, str], player_id: int, override_season: Optional[int]) -> Tuple[int, int]:
    """
    Logique robuste :
    - si season saisie -> /players?id=&season= pour trouver team
    - sinon -> /players/seasons pour trouver la plus récente, puis /players?id=&season=
    - en dernier recours -> lève une erreur claire
    """
    if override_season:
        team_id, _ = player_team_from_players_endpoint(headers, player_id, int(override_season))
        return team_id, int(override_season)
    seasons = get_player_seasons(headers, player_id)
    if not seasons:
        raise RuntimeError("Aucune saison disponible pour ce joueur.")
    best = max(seasons)
    team_id, _ = player_team_from_players_endpoint(headers, player_id, best)
    return team_id, best

# =========================
# --- Données par fixture
# =========================
def get_team_fixtures(headers: Dict[str, str], team_id: int, season: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures", headers, params={"team": team_id, "season": season})
    return [fx for fx in (data.get("response", []) or []) if isinstance(fx, dict)]

def get_fixture_players(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/players", headers, params={"fixture": fixture_id})
    return [x for x in (data.get("response", []) or []) if isinstance(x, dict)]

def get_fixture_events(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/events", headers, params={"fixture": fixture_id})
    return [e for e in (data.get("response", []) or []) if isinstance(e, dict)]

def get_fixture_lineups(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/lineups", headers, params={"fixture": fixture_id})
    return [x for x in (data.get("response", []) or []) if isinstance(x, dict)]

def get_fixture_odds(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    try:
        data = http_get("odds", headers, params={"fixture": fixture_id})
        return [x for x in (data.get("response", []) or []) if isinstance(x, dict)]
    except Exception:
        return []

def extract_player_minutes_goals(fixture_players_payload: List[Dict[str, Any]], player_id: int) -> Tuple[int, int]:
    minutes, goals = 0, 0
    for teamblock in fixture_players_payload:
        players = teamblock.get("players") or []
        if not isinstance(players, list):
            continue
        for p in players:
            if not isinstance(p, dict):
                continue
            pdata = p.get("player") or {}
            pid = pdata.get("id")
            if pid == player_id:
                stats_list = p.get("statistics") or []
                stats = stats_list[0] if stats_list and isinstance(stats_list[0], dict) else {}
                minutes = int((stats.get("games") or {}).get("minutes", 0) or 0) if isinstance(stats.get("games"), dict) else 0
                goals   = int((stats.get("goals") or {}).get("total", 0) or 0) if isinstance(stats.get("goals"), dict) else 0
                return minutes, goals
    return minutes, goals

def player_took_penalty(events: List[Dict[str, Any]], player_id: int) -> bool:
    for e in events:
        etype = str(e.get("type") or "").lower()
        detail = str(e.get("detail") or "").lower()
        pid = (e.get("player") or {}).get("id")
        if pid == player_id and (
            (etype == "goal" and "penalty" in detail) or etype == "penalty"
        ):
            return True
    return False

def extract_opponent_price(odds_payload: List[Dict[str, Any]], opponent_side: str) -> Optional[float]:
    opp = opponent_side.lower()  # "home"/"away"
    for o in odds_payload:
        for book in (o.get("bookmakers") or []):
            if not isinstance(book, dict):
                continue
            for bet in (book.get("bets") or []):
                if not isinstance(bet, dict):
                    continue
                name = str(bet.get("name") or "").lower()
                if name in {"match winner", "1x2", "winner", "match result"}:
                    for v in (bet.get("values") or []):
                        if not isinstance(v, dict):
                            continue
                        if str(v.get("value") or "").lower() == opp:
                            try:
                                return float(v.get("odd"))
                            except Exception:
                                pass
    return None

def get_top_assisters_ids(headers: Dict[str, str], league_id: int, season: int) -> List[int]:
    try:
        data = http_get("players/topassists", headers, params={"league": league_id, "season": season})
        ids: List[int] = []
        for r in (data.get("response", []) or []):
            if not isinstance(r, dict):
                continue
            pid = (r.get("player") or {}).get("id")
            if isinstance(pid, int):
                ids.append(pid)
        return ids
    except Exception:
        return []

def any_of_players_in_lineups(lineups: List[Dict[str, Any]], player_ids: List[int]) -> bool:
    present = set()
    for team in lineups:
        for blk in ("startXI", "substitutes"):
            for p in (team.get(blk) or []):
                if not isinstance(p, dict):
                    continue
                pid = (p.get("player") or {}).get("id")
                if isinstance(pid, int):
                    present.add(pid)
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
    secret_key = st.secrets.get("API_KEY") if hasattr(st, "secrets") else None
    api_key = st.text_input("API Key", value=secret_key or "", type="password")
    season_in = st.text_input("Saison (YYYY) — optionnel", value="")
    team_hint = st.text_input("Équipe (optionnel)", value="", help="Aide à désambigüiser la recherche joueur.")
    player_query = st.text_input("🔎 Joueur (ex: Barnabás Varga)", value="")
    run_btn = st.button("▶️ Rechercher & extraire")

def export_csv(df: pd.DataFrame, filename: str):
    st.download_button("📥 Télécharger CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")

if run_btn:
    try:
        if not api_key.strip():
            st.error("Merci de renseigner la clé API."); st.stop()
        if not player_query.strip():
            st.error("Merci de saisir un nom de joueur."); st.stop()

        headers = api_headers(provider, api_key.strip())

        # (Info) statut
        try:
            status = http_get("status", headers)
            sub = status.get("response", {}).get("subscription", {})
            st.success(f"Plan: {sub.get('plan')} | actif: {sub.get('active')}")
        except Exception:
            st.info("Info: /status indisponible (non bloquant).")

        # 1) Trouver le joueur (équipe si fournie -> effectif, sinon profils)
        chosen_player_id: Optional[int] = None
        if team_hint.strip():
            tid = search_team_id(headers, team_hint)
            if tid:
                cands = list_squad_candidates(headers, tid, player_query)
                if len(cands) == 1:
                    chosen_player_id = cands[0]["id"]
                elif len(cands) > 1:
                    st.subheader("Plusieurs joueurs trouvés :")
                    label_map = {f"{c['name']} — {c.get('position','?')} — id:{c['id']}": c["id"] for c in cands}
                    choice = st.selectbox("Sélectionne le joueur exact", list(label_map.keys()))
                    chosen_player_id = int(label_map[choice])
                else:
                    st.warning("Aucun joueur correspondant dans l’effectif indiqué. On tente une recherche globale…")

        if not chosen_player_id:
            profs = profiles_by_lastname(headers, player_query)
            if not profs:
                raise RuntimeError("Aucun profil joueur trouvé.")
            options = []
            for p in profs[:50]:
                pid = p.get("id")
                if not isinstance(pid, int):
                    continue
                nm = p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip()
                nat = p.get("nationality") or "?"
                by  = (p.get("birth") or {}).get("date") if isinstance(p.get("birth"), dict) else "?"
                options.append((f"{nm} — {nat} — id:{pid}", pid))
            if not options:
                raise RuntimeError("Aucun profil exploitable (id manquant).")
            label = st.selectbox("Sélectionne le joueur", [o[0] for o in options])
            chosen_player_id = int(dict(options)[label])

        st.write(f"**Joueur choisi** → id : `{chosen_player_id}`")

        # 2) Saison & équipe via /players (robuste)
        override_season = int(season_in) if season_in.strip().isdigit() else None
        team_id, season = resolve_team_and_season(headers, chosen_player_id, override_season)
        st.write(f"**Équipe** id `{team_id}` — **Saison** `{season}`")

        # 3) Fixtures & extraction
        fixtures = get_team_fixtures(headers, team_id, season)
        st.write(f"{len(fixtures)} fixtures trouvées.")
        prog = st.progress(0.0)

        rows = []
        top_assists_cache: Dict[Tuple[int,int], List[int]] = {}
        total = max(1, len(fixtures))

        for idx, fx in enumerate(fixtures, start=1):
            fixture = fx.get("fixture") if isinstance(fx.get("fixture"), dict) else {}
            league  = fx.get("league")  if isinstance(fx.get("league"),  dict) else {}
            teams   = fx.get("teams")   if isinstance(fx.get("teams"),   dict) else {}

            fixture_id = fixture.get("id")
            date_iso   = str(fixture.get("date") or "")[:10]
            league_id  = league.get("id")
            round_str  = league.get("round") or ""

            home = teams.get("home") if isinstance(teams.get("home"), dict) else {}
            away = teams.get("away") if isinstance(teams.get("away"), dict) else {}
            home_id, away_id = home.get("id"), away.get("id")
            home_name, away_name = home.get("name"), away.get("name")

            if not (isinstance(fixture_id, int) and isinstance(home_id, int) and isinstance(away_id, int)):
                prog.progress(idx/total); continue

            if team_id == home_id:
                dom_ext = "D"; opponent_name = away_name; opponent_side = "away"
            else:
                dom_ext = "E"; opponent_name = home_name; opponent_side = "home"

            # stats joueur
            fps = get_fixture_players(headers, fixture_id)
            minutes, goals = extract_player_minutes_goals(fps, chosen_player_id)
            scored = goals > 0

            # penalty ?
            evs = get_fixture_events(headers, fixture_id)
            took_pen = player_took_penalty(evs, chosen_player_id)

            # lineups
            lineups = get_fixture_lineups(headers, fixture_id)

            # odds (cote adversaire 1X2)
            odds = get_fixture_odds(headers, fixture_id)
            opp_price = extract_opponent_price(odds, opponent_side)

            # top assists présent ?
            key_assister_present = 0
            if isinstance(league_id, int):
                key = (league_id, season)
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
