import time
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set

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
    data.setdefault("errors", {})
    data.setdefault("response", [])
    return data

def yesno(b: bool) -> int:
    return 1 if b else 0

# =========================
# --- Recherche joueur
# =========================
def _normalize_profiles_list(resp: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for x in resp:
        if isinstance(x, dict):
            if "id" in x:
                out.append(x)
            elif "player" in x and isinstance(x["player"], dict) and "id" in x["player"]:
                merged = dict(x["player"])
                if "name" not in merged and "name" in x:
                    merged["name"] = x["name"]
                out.append(merged)
    return out

def _build_search_terms(player_query: str) -> List[str]:
    q = norm(player_query)
    toks = [t for t in q.split() if len(t) >= 2]
    if not toks and len(q) >= 2:
        toks = [q[:2]]
    terms = list(dict.fromkeys(toks + ([q] if len(q) >= 2 else [])))
    return terms[:6]

def profiles_search_smart(headers: Dict[str, str], player_query: str) -> List[Dict[str, Any]]:
    terms = _build_search_terms(player_query)
    seen: Set[int] = set()
    bag: List[Dict[str, Any]] = []
    for t in terms:
        data = http_get("players/profiles", headers, params={"search": t})
        prof = _normalize_profiles_list(data["response"])
        for p in prof:
            pid = p.get("id")
            if isinstance(pid, int) and pid not in seen:
                seen.add(pid); bag.append(p)
        time.sleep(0.08)
    qtokens = [t for t in norm(player_query).split() if t]
    scored = []
    for p in bag:
        cand = p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip()
        cn = norm(cand)
        score = sum(tok in cn for tok in qtokens) + (2 if norm(player_query) == cn else 0)
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], norm(x[1].get("name") or "")))
    return [p for _, p in scored][:50]

def profiles_bruteforce(headers: Dict[str, str], player_query: str, pages: int = 3) -> List[Dict[str, Any]]:
    needle = norm(player_query)
    bag: List[Dict[str, Any]] = []
    for page in range(1, max(1, pages) + 1):
        data = http_get("players/profiles", headers, params={"page": page})
        for p in _normalize_profiles_list(data["response"]):
            name = p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip()
            if needle and needle in norm(name):
                bag.append(p)
        time.sleep(0.08)
    seen: Set[int] = set(); uniq = []
    for p in bag:
        pid = p.get("id")
        if isinstance(pid, int) and pid not in seen:
            seen.add(pid); uniq.append(p)
    return uniq[:50]

# =========================
# --- Recherche pays / équipes (ROBUSTE)
# =========================
def countries_lookup(headers: Dict[str, str], hint: str) -> List[Dict[str, Any]]:
    """
    Retourne des pays probables via /countries en testant name, code, search.
    """
    h = hint.strip()
    if not h:
        return []
    tries = [
        {"name": h},
        {"code": h.upper()},
        {"search": h[:3]} if len(h) >= 3 else None,
        {"name": unicodedata.normalize("NFKD", h).encode("ascii", "ignore").decode()}  # sans accents
    ]
    out = []
    for params in [t for t in tries if t]:
        data = http_get("countries", headers, params=params)
        for c in data["response"]:
            if isinstance(c, dict):
                name = c.get("name"); code = c.get("code")
                if name:
                    out.append({"name": name, "code": code})
        time.sleep(0.06)
    # dédoublonne par nom
    seen: Set[str] = set(); uniq = []
    for c in out:
        key = norm(c["name"])
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq

def teams_by_country(headers: Dict[str, str], country_name: str) -> List[Dict[str, Any]]:
    data = http_get("teams", headers, params={"country": country_name})
    out = []
    for r in data["response"]:
        team = r.get("team") or {}
        if isinstance(team, dict) and isinstance(team.get("id"), int):
            out.append({
                "id": team.get("id"),
                "name": team.get("name"),
                "national": team.get("national")
            })
    return out

def search_teams_robust(headers: Dict[str, str], hint: str, prefer_national: bool = True) -> List[Dict[str, Any]]:
    """
    1) /teams?search=hint (avec & sans accents)
    2) /countries -> /teams?country=NomDuPays
    3) filtre national si demandé
    """
    out: List[Dict[str, Any]] = []
    seen: Set[int] = set()

    # 1) direct search
    for term in {hint, unicodedata.normalize("NFKD", hint).encode("ascii", "ignore").decode()}:
        if term.strip():
            data = http_get("teams", headers, params={"search": term})
            for r in data["response"]:
                team = r.get("team") or {}
                if isinstance(team, dict) and isinstance(team.get("id"), int):
                    tid = team["id"]
                    if tid not in seen:
                        seen.add(tid)
                        out.append({
                            "id": tid, "name": team.get("name"),
                            "national": team.get("national")
                        })
            time.sleep(0.06)

    # 2) via pays
    countries = countries_lookup(headers, hint)
    for c in countries:
        for t in teams_by_country(headers, c["name"]):
            tid = t["id"]
            if tid not in seen:
                seen.add(tid); out.append(t)
        time.sleep(0.06)

    # 3) filtre national si demandé
    if prefer_national:
        # garde les sélections si présentes, sinon retourne tout
        nationals = [t for t in out if t.get("national")]
        if nationals:
            out = nationals

    # tri: sélections d'abord, puis alpha
    out = sorted(out, key=lambda t: (0 if t.get("national") else 1, norm(t.get("name") or "")))
    return out

# =========================
# --- Fallback équipes depuis stats
# =========================
def get_player_seasons(headers: Dict[str, str], player_id: int) -> List[int]:
    data = http_get("players/seasons", headers, params={"player": player_id})
    seasons = data.get("response", []) or []
    return [int(s) for s in seasons if isinstance(s, int)]

def teams_from_players_stats(headers: Dict[str, str], player_id: int, seasons: List[int], limit: int = 5) -> List[Dict[str, Any]]:
    teams: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    for y in sorted(seasons, reverse=True)[:max(1, limit)]:
        data = http_get("players", headers, params={"id": player_id, "season": y})
        for rec in data["response"]:
            for st in rec.get("statistics", []) or []:
                team = st.get("team") or {}
                tid, tname = team.get("id"), team.get("name")
                if isinstance(tid, int) and tid not in seen:
                    seen.add(tid)
                    teams.append({"id": tid, "name": tname, "national": False})
        time.sleep(0.08)
    return sorted(teams, key=lambda t: norm(t["name"] or ""))

# =========================
# --- Fixtures & extraction
# =========================
def fixtures_by_team_last(headers: Dict[str, str], team_id: int, n: int = 50) -> List[Dict[str, Any]]:
    data = http_get("fixtures", headers, params={"team": team_id, "last": min(max(n,1), 50)})
    return [fx for fx in (data["response"] or []) if isinstance(fx, dict)]

def fixtures_by_team_range(headers: Dict[str, str], team_id: int, date_from: str, date_to: str) -> List[Dict[str, Any]]:
    data = http_get("fixtures", headers, params={"team": team_id, "from": date_from, "to": date_to})
    return [fx for fx in (data["response"] or []) if isinstance(fx, dict)]

def get_fixture_players(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/players", headers, params={"fixture": fixture_id})
    return [x for x in (data["response"] or []) if isinstance(x, dict)]

def get_fixture_events(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/events", headers, params={"fixture": fixture_id})
    return [e for e in (data["response"] or []) if isinstance(e, dict)]

def get_fixture_lineups(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures/lineups", headers, params={"fixture": fixture_id})
    return [x for x in (data["response"] or []) if isinstance(x, dict)]

def get_fixture_odds(headers: Dict[str, str], fixture_id: int) -> List[Dict[str, Any]]:
    try:
        data = http_get("odds", headers, params={"fixture": fixture_id})
        return [x for x in (data["response"] or []) if isinstance(x, dict)]
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
        if pid == player_id and ((etype == "goal" and "penalty" in detail) or etype == "penalty"):
            return True
    return False

def extract_opponent_price(odds_payload: List[Dict[str, Any]], opponent_side: str) -> Optional[float]:
    opp = opponent_side.lower()
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

def get_top_assisters_ids(headers: Dict[str, str], league_id: Optional[int], season: Optional[int]) -> List[int]:
    if not (isinstance(league_id, int) and isinstance(season, int)):
        return []
    try:
        data = http_get("players/topassists", headers, params={"league": league_id, "season": season})
        ids: List[int] = []
        for r in (data["response"] or []):
            pid = (r.get("player") or {}).get("id")
            if isinstance(pid, int):
                ids.append(pid)
        return ids
    except Exception:
        return []

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
    player_query = st.text_input("🔎 Joueur (ex: Kylian Mbappe)", value="")
    player_id_override = st.text_input("ID joueur (optionnel)", value="")
    team_hint = st.text_input("Indice équipe (pays/club/code ex: France / FR / FRA)", value="")
    months_back = st.slider("Fenêtre glissante (mois)", 6, 60, 36)
    run_btn = st.button("▶️ Rechercher & extraire")

def export_csv(df: pd.DataFrame, filename: str):
    st.download_button("📥 Télécharger CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")

if run_btn:
    try:
        if not api_key.strip():
            st.error("Merci de renseigner la clé API."); st.stop()

        headers = api_headers(provider, api_key.strip())

        # (Info) statut
        try:
            status = http_get("status", headers)
            sub = status.get("response", {}).get("subscription", {})
            st.success(f"Plan: {sub.get('plan')} | actif: {sub.get('active')}")
        except Exception:
            st.info("Info: /status indisponible (non bloquant).")

        # --- 1) Choix du joueur (ID direct / recherche / bruteforce)
        chosen_player: Optional[Dict[str, Any]] = None

        if player_id_override.strip().isdigit():
            pid = int(player_id_override.strip())
            prof = http_get("players/profiles", headers, params={"player": pid})
            resp = _normalize_profiles_list(prof["response"])
            chosen_player = resp[0] if resp else {"id": pid, "name": f"Player {pid}"}

        if not chosen_player:
            if not player_query.strip():
                st.error("Saisis un nom de joueur ou un ID."); st.stop()
            profs = profiles_search_smart(headers, player_query)
            if not profs:
                st.info("Recherche étendue (bruteforce) pages 1..3…")
                profs = profiles_bruteforce(headers, player_query, pages=3)
            if not profs:
                raise RuntimeError("Aucun profil joueur trouvé après recherche + bruteforce.")
            opts = []
            for p in profs:
                pid = p.get("id")
                nm = p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip()
                nat = p.get("nationality") or "?"
                opts.append((f"{nm} — {nat} — id:{pid}", pid, p))
            label = st.selectbox("Sélectionne le joueur", [o[0] for o in opts])
            chosen_player = next(p for (lab, pid, p) in opts if lab == label)

        player_id = int(chosen_player.get("id"))
        player_name = chosen_player.get("name") or f"{chosen_player.get('firstname','')} {chosen_player.get('lastname','')}".strip()
        nationality = (chosen_player.get("nationality") or "").strip()
        st.write(f"**Joueur choisi** → {player_name} (id: `{player_id}`)")

        # --- 2) Équipes du joueur via /players/teams
        team_candidates: List[Dict[str, Any]] = []
        try:
            pteams = http_get("players/teams", headers, params={"player": player_id})
            for r in pteams.get("response", []) or []:
                team = r.get("team") or {}
                if isinstance(team, dict) and isinstance(team.get("id"), int):
                    team_candidates.append({
                        "id": team.get("id"),
                        "name": team.get("name"),
                        "national": team.get("national"),
                    })
        except Exception:
            pass

        # --- 3) Fallback recherche d’équipe robuste à partir de l’indice saisi
        if team_hint.strip():
            robust = search_teams_robust(headers, team_hint, prefer_national=True)
            for t in robust:
                if all(t["id"] != x.get("id") for x in team_candidates):
                    team_candidates.append(t)

        # --- 4) Encore rien ? On propose automatiquement la sélection du pays de nationalité
        if not team_candidates and nationality:
            robust_nat = search_teams_robust(headers, nationality, prefer_national=True)
            for t in robust_nat:
                if all(t["id"] != x.get("id") for x in team_candidates):
                    team_candidates.append(t)

        # --- 5) Toujours rien ? On reconstitue des clubs récents via /players -> statistics
        if not team_candidates:
            seasons = get_player_seasons(headers, player_id)
            clubs = teams_from_players_stats(headers, player_id, seasons, limit=5)
            team_candidates.extend(clubs)

        if not team_candidates:
            raise RuntimeError("Aucune équipe candidate trouvée (même avec recherche robuste et fallback). Essaie un autre indice (pays/club/code).")

        # tri final: sélections d’abord
        team_candidates = sorted(team_candidates, key=lambda t: (0 if t.get("national") else 1, norm(t.get("name") or "")))
        tlabels = [f"{t['name']} — id:{t['id']} — {'Sélection' if t.get('national') else 'Club'}" for t in team_candidates]
        tchoice = st.selectbox("Choisis l’équipe", tlabels)
        team_id = int(tchoice.split("id:")[1].split(" ")[0])

        # --- 6) Fixtures (last50 puis fenêtre glissante)
        fixtures = fixtures_by_team_last(headers, team_id, n=50)
        used_note = "last50"
        if len(fixtures) == 0:
            to_date = datetime.utcnow().date()
            from_date = to_date - timedelta(days=months_back*30)
            fixtures = fixtures_by_team_range(headers, team_id, from_date.isoformat(), to_date.isoformat())
            used_note = f"range_{from_date.isoformat()}_to_{to_date.isoformat()}"

        if len(fixtures) == 0:
            st.error("Aucune fixture trouvée. Augmente la fenêtre (mois) ou choisis une autre équipe.")
            st.stop()

        st.write(f"{len(fixtures)} fixtures trouvées. Source: {used_note}")
        prog = st.progress(0.0)

        # --- 7) Extraction
        rows = []
        top_assists_cache: Dict[Tuple[int, Optional[int]], List[int]] = {}
        total = max(1, len(fixtures))

        for idx, fx in enumerate(fixtures, start=1):
            fixture = fx.get("fixture") if isinstance(fx.get("fixture"), dict) else {}
            league  = fx.get("league")  if isinstance(fx.get("league"),  dict) else {}
            teams   = fx.get("teams")   if isinstance(fx.get("teams"),   dict) else {}

            fixture_id = fixture.get("id")
            date_iso   = str(fixture.get("date") or "")[:10]
            league_id  = league.get("id")
            round_str  = league.get("round") or ""
            season_fx  = league.get("season")

            home = teams.get("home") if isinstance(teams.get("home"), dict) else {}
            away = teams.get("away") if isinstance(teams.get("away"), dict) else {}
            home_id, away_id = home.get("id"), away.get("id")
            home_name, away_name = home.get("name"), away.get("name")

            if not (isinstance(fixture_id, int) and isinstance(home_id, int) and isinstance(away_id, int)):
                prog.progress(idx/total); continue

            if team_id == home_id:
                dom_ext = "D"; opponent_name = away_name; opponent_side = "away"
            elif team_id == away_id:
                dom_ext = "E"; opponent_name = home_name; opponent_side = "home"
            else:
                dom_ext = "?"; opponent_name = home_name or away_name; opponent_side = "away"

            fps = get_fixture_players(headers, fixture_id)
            minutes, goals = extract_player_minutes_goals(fps, player_id)

            evs = get_fixture_events(headers, fixture_id)
            took_pen = player_took_penalty(evs, player_id)

            lineups = get_fixture_lineups(headers, fixture_id)

            odds = get_fixture_odds(headers, fixture_id)
            opp_price = extract_opponent_price(odds, opponent_side)

            key_assister_present = 0
            if isinstance(league_id, int) and isinstance(season_fx, int):
                key = (league_id, season_fx)
                if key not in top_assists_cache:
                    top_assists_cache[key] = get_top_assisters_ids(headers, league_id, season_fx)
                if top_assists_cache[key]:
                    key_assister_present = yesno(any_of_players_in_lineups(lineups, top_assists_cache[key]))

            important = yesno(is_important_round(round_str))

            rows.append({
                "date": date_iso,
                "adversaire": opponent_name,
                "dom_ext": dom_ext,
                "minutes": minutes,
                "but": int(goals > 0),
                "tire_pen": int(took_pen),
                "cote_adversaire_1x2": opp_price,
                "passeur_principal_present": key_assister_present,
                "important": important,
                "fixture_id": fixture_id,
                "league_id": league_id,
                "round": round_str,
            })

            time.sleep(0.05)
            prog.progress(min(1.0, idx/total))

        df = pd.DataFrame(rows)
        st.subheader("Résultats")
        if df.empty:
            st.warning("Aucune ligne à afficher (DF vide)."); st.stop()
        if "date" in df.columns:
            df = df.sort_values(["date", "fixture_id"] if "fixture_id" in df.columns else ["date"])
        st.dataframe(df, use_container_width=True)
        export_csv(df, filename=f"player_{player_id}.csv")

    except Exception as e:
        st.error(f"Erreur : {e}")
