import time
import unicodedata
from datetime import datetime, timedelta
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
def search_teams(headers: Dict[str, str], q: str) -> List[Dict[str, Any]]:
    data = http_get("teams", headers, params={"search": q})
    out = []
    for r in data.get("response", []) or []:
        if not isinstance(r, dict):
            continue
        team = r.get("team") or {}
        country = r.get("country") or {}
        if isinstance(team, dict):
            out.append({
                "id": team.get("id"),
                "name": team.get("name"),
                "code": team.get("code"),
                "country": country if isinstance(country, str) else r.get("country"),
                "national": team.get("national"),  # True pour sélections
            })
    # dédupe, garde ordonné
    seen = set()
    uniq = []
    for t in out:
        tid = t.get("id")
        if isinstance(tid, int) and tid not in seen:
            seen.add(tid); uniq.append(t)
    return uniq

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
                out.append(x); continue
            if "player" in x and isinstance(x["player"], dict) and "id" in x["player"]:
                merged = dict(x["player"])
                if "name" not in merged and "name" in x:
                    merged["name"] = x["name"]
                out.append(merged); continue
    return out

def _build_search_terms(player_query: str) -> List[str]:
    q = norm(player_query)
    toks = [t for t in q.split() if len(t) >= 3]
    if not toks and len(q) >= 3:
        toks = [q[:3]]
    terms = list(dict.fromkeys(toks + ([q] if len(q) >= 3 else [])))
    return terms[:5]

def profiles_search_smart(headers: Dict[str, str], player_query: str) -> List[Dict[str, Any]]:
    terms = _build_search_terms(player_query)
    seen: set = set()
    bag: List[Dict[str, Any]] = []
    for t in terms:
        data = http_get("players/profiles", headers, params={"search": t})
        prof = _normalize_profiles_list(data.get("response", []) or [])
        for p in prof:
            pid = p.get("id")
            if isinstance(pid, int) and pid not in seen:
                seen.add(pid); bag.append(p)
        time.sleep(0.1)
    qtokens = [t for t in norm(player_query).split() if t]
    scored = []
    for p in bag:
        cand = p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip()
        cn = norm(cand)
        score = sum(tok in cn for tok in qtokens)
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], norm(x[1].get("name") or "")))
    return [p for _, p in scored][:50]

# =========================
# --- Fixtures (robustes, multi-stratégies)
# =========================
def fixtures_by_team_season(headers: Dict[str, str], team_id: int, season: int) -> List[Dict[str, Any]]:
    data = http_get("fixtures", headers, params={"team": team_id, "season": season})
    return [fx for fx in (data.get("response", []) or []) if isinstance(fx, dict)]

def fixtures_by_team_last(headers: Dict[str, str], team_id: int, n: int = 50) -> List[Dict[str, Any]]:
    data = http_get("fixtures", headers, params={"team": team_id, "last": min(max(n,1), 50)})
    return [fx for fx in (data.get("response", []) or []) if isinstance(fx, dict)]

def fixtures_by_team_range(headers: Dict[str, str], team_id: int, date_from: str, date_to: str) -> List[Dict[str, Any]]:
    data = http_get("fixtures", headers, params={"team": team_id, "from": date_from, "to": date_to})
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

def get_top_assisters_ids(headers: Dict[str, str], league_id: int, season: Optional[int]) -> List[int]:
    if not isinstance(league_id, int) or season is None:
        return []
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
    api_key = st.text_input("API Key", value=secret_key or "", type="password",
                            help="Astuce : ajoute API_KEY dans Settings > Secrets sur Streamlit Cloud.")
    player_query = st.text_input("🔎 Joueur (ex: Kylian Mbappé)", value="")
    team_hint = st.text_input("Équipe (ex: France ou Real Madrid)", value="")
    season_in = st.text_input("Saison (YYYY, optionnel)", value="")
    months_back = st.slider("Fallback fenêtre (mois) si aucune saison ne marche", 6, 48, 36)
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

        # --- 1) Trouver le joueur
        chosen_player_id: Optional[int] = None
        # On essaie d'abord via l'équipe si fournie
        chosen_team_id: Optional[int] = None

        # Choix de l'équipe si hint fourni (on laisse sélectionner)
        if team_hint.strip():
            team_candidates = search_teams(headers, team_hint)
            if not team_candidates:
                st.warning("Aucune équipe trouvée pour ce libellé. On continue sans filtre équipe.")
            elif len(team_candidates) == 1:
                chosen_team_id = team_candidates[0]["id"]
                st.write(f"**Équipe choisie** → {team_candidates[0]['name']} (id {chosen_team_id})")
            else:
                labels = [f"{t['name']} — id:{t['id']} — national:{t.get('national')}" for t in team_candidates]
                lab = st.selectbox("Plusieurs équipes trouvées : sélectionne la bonne", labels)
                chosen_team_id = int(lab.split("id:")[1].split(" ")[0])
                st.write(f"**Équipe choisie** → id : `{chosen_team_id}`")

        # a) si équipe connue → chercher le joueur dans l’effectif
        if chosen_team_id:
            cands = list_squad_candidates(headers, chosen_team_id, player_query)
            if len(cands) == 1:
                chosen_player_id = cands[0]["id"]
            elif len(cands) > 1:
                st.subheader("Homonymes dans l’effectif :")
                label_map = {f"{c['name']} — {c.get('position','?')} — id:{c['id']}": c["id"] for c in cands}
                choice = st.selectbox("Sélectionne le joueur exact", list(label_map.keys()))
                chosen_player_id = int(label_map[choice])
            else:
                st.info("Joueur non trouvé dans cet effectif. Recherche globale…")

        # b) recherche globale /players/profiles
        if not chosen_player_id:
            profs = profiles_search_smart(headers, player_query)
            if not profs:
                raise RuntimeError("Aucun profil joueur trouvé. Corrige l’orthographe ou précise l’équipe.")
            options = []
            for p in profs[:50]:
                pid = p.get("id")
                if not isinstance(pid, int):
                    continue
                nm = p.get("name") or f"{p.get('firstname','')} {p.get('lastname','')}".strip()
                nat = p.get("nationality") or "?"
                by  = (p.get("birth") or {}).get("date") if isinstance(p.get("birth"), dict) else "?"
                options.append((f"{nm} — {nat} — id:{pid}", pid))
            label = st.selectbox("Sélectionne le joueur", [o[0] for o in options])
            chosen_player_id = int(dict(options)[label])

        st.write(f"**Joueur choisi** → id : `{chosen_player_id}`")

        # --- 2) Déterminer l’équipe finale (si non choisie) :
        #     - Si une équipe a été choisie via la sélection ci-dessus, on la garde.
        #     - Sinon, on laisse l’utilisateur taper un hint ; s’il est absent, on s’arrêtera si 0 fixtures.
        if not chosen_team_id and team_hint.strip():
            # si l'utilisateur a tapé une équipe mais n'a pas choisi (pas de multiples), retentons
            teams_retry = search_teams(headers, team_hint)
            if teams_retry:
                chosen_team_id = teams_retry[0]["id"]
                st.write(f"**Équipe déduite** → id : `{chosen_team_id}`")

        # --- 3) Chercher les fixtures (multi-stratégies)
        fixtures: List[Dict[str, Any]] = []
        used_note = ""
        used_season: Optional[int] = None

        # Si on a une saison saisie et une équipe → tenter par saison
        if chosen_team_id and season_in.strip().isdigit():
            y = int(season_in.strip())
            fixtures = fixtures_by_team_season(headers, chosen_team_id, y)
            used_season = y
            used_note = "season_exact"
            if len(fixtures) == 0:
                st.warning("0 fixtures sur la saison saisie. On tente un fallback (derniers matchs).")

        # Fallback 1 : derniers N matchs (sans saison)
        if chosen_team_id and len(fixtures) == 0:
            fixtures = fixtures_by_team_last(headers, chosen_team_id, n=50)
            used_note = used_note or "last50"

        # Fallback 2 : fenêtre glissante (ex. 36 mois)
        if chosen_team_id and len(fixtures) == 0:
            to_date = datetime.utcnow().date()
            from_date = to_date - timedelta(days=months_back*30)
            fixtures = fixtures_by_team_range(headers, chosen_team_id, from_date.isoformat(), to_date.isoformat())
            used_note = used_note or f"range_{from_date.isoformat()}_to_{to_date.isoformat()}"

        # Si on n’a toujours rien et pas d’équipe fiable => on ne peut pas continuer
        if len(fixtures) == 0:
            st.error("Aucune fixture trouvée après tous les fallbacks. Vérifie que l’équipe sélectionnée est la bonne (ex: France A) ou essaye un club.")
            st.stop()

        st.write(f"{len(fixtures)} fixtures trouvées. Source: {used_note or 'inconnue'}")
        prog = st.progress(0.0)

        # --- 4) Extraction par fixture
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
            season_fx  = league.get("season")  # utile pour topassists si dispo

            home = teams.get("home") if isinstance(teams.get("home"), dict) else {}
            away = teams.get("away") if isinstance(teams.get("away"), dict) else {}
            home_id, away_id = home.get("id"), away.get("id")
            home_name, away_name = home.get("name"), away.get("name")

            # si aucune équipe choisie avant, on devine via présence du joueur dans lineups plus bas ; pour l’instant on saute
            if not isinstance(fixture_id, int) or not isinstance(home_id, int) or not isinstance(away_id, int):
                prog.progress(idx/total); continue
            if not chosen_team_id:
                # si pas d'équipe, on ne peut pas déterminer D/E -> on continue quand même, mais on marquera 'dom_ext'='?'
                team_id = None
                dom_ext = "?"
                opponent_name = home_name or away_name
                opponent_side = "home"
            else:
                team_id = chosen_team_id
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
            opp_price = extract_opponent_price(odds, opponent_side if chosen_team_id else "away")

            # top assists présent ? (si ligue + saison connus, sinon on met 0)
            key_assister_present = 0
            if isinstance(league_id, int):
                key = (league_id, season_fx if isinstance(season_fx, int) else used_season)
                if key not in top_assists_cache:
                    top_assists_cache[key] = get_top_assisters_ids(headers, league_id, key[1])
                if top_assists_cache[key]:
                    key_assister_present = yesno(any_of_players_in_lineups(lineups, top_assists_cache[key]))

            # importance KO ?
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
                "round": round_str,
            })

            time.sleep(0.1)
            prog.progress(min(1.0, idx/total))

        df = pd.DataFrame(rows)
        st.subheader("Résultats")

        if df.empty:
            st.warning("Aucune ligne à afficher (DF vide).")
            st.stop()

        # Tri si colonnes présentes
        sort_cols = [c for c in ["date", "fixture_id"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        st.dataframe(df, use_container_width=True)
        export_csv(df, filename=f"player_{chosen_player_id}.csv")

    except Exception as e:
        st.error(f"Erreur : {e}")
