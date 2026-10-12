"""Thin API-Football (api-sports.io v3) client with an on-disk cache so the free
plan's 100 requests/day is enough for the daily run."""
import hashlib
import json
import time
from collections import defaultdict
import requests

from .config import API_KEY, CACHE_DIR, LEAGUES

BASE = "https://v3.football.api-sports.io"


def _get(path, params, ttl):
    key = hashlib.md5((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    f = CACHE_DIR / f"{key}.json"
    if f.exists() and time.time() - f.stat().st_mtime < ttl:
        return json.loads(f.read_text())
    r = requests.get(BASE + path, params=params, headers={"x-apisports-key": API_KEY}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(f"API-Football error: {data['errors']}")
    f.write_text(json.dumps(data))
    return data


def _norm_fixture(x):
    return {
        "id": x["fixture"]["id"],
        "date": x["fixture"]["date"][:10],
        "kickoff": x["fixture"]["date"],
        "status": x["fixture"]["status"]["short"],
        "league_id": x["league"]["id"],
        "league": x["league"]["name"],
        "home_id": x["teams"]["home"]["id"], "home": x["teams"]["home"]["name"],
        "away_id": x["teams"]["away"]["id"], "away": x["teams"]["away"]["name"],
        "hg": x["goals"]["home"], "ag": x["goals"]["away"],
        "ht_hg": x["score"]["halftime"]["home"], "ht_ag": x["score"]["halftime"]["away"],
    }


def finished_matches(league_id, season, ttl=6 * 3600):
    d = _get("/fixtures", {"league": league_id, "season": season, "status": "FT-AET-PEN"}, ttl)
    return [_norm_fixture(x) for x in d["response"] if x["goals"]["home"] is not None]


def fixtures_on(date_str, ttl=1800):
    d = _get("/fixtures", {"date": date_str}, ttl)
    out = [_norm_fixture(x) for x in d["response"] if x["league"]["id"] in LEAGUES]
    return [m for m in out if m["status"] in ("NS", "TBD", "PST")]


def injuries(league_id, season, date_str, ttl=3 * 3600):
    d = _get("/injuries", {"league": league_id, "season": season, "date": date_str}, ttl)
    res = defaultdict(list)
    for x in d["response"]:
        res[x["team"]["id"]].append({
            "player_id": x["player"]["id"], "name": x["player"]["name"],
            "type": x["player"].get("type"), "reason": x["player"].get("reason"),
        })
    return res


def top_contributors(league_id, season, ttl=24 * 3600):
    """{team_id: {player_id: goals+assists}} from the league's top scorer/assist tables."""
    out = defaultdict(dict)
    for ep in ("topscorers", "topassists"):
        try:
            d = _get(f"/players/{ep}", {"league": league_id, "season": season}, ttl)
        except Exception:
            continue
        for x in d["response"]:
            p = x["player"]["id"]
            st = x["statistics"][0]
            tid = st["team"]["id"]
            g = (st["goals"].get("total") or 0)
            a = (st["goals"].get("assists") or 0)
            out[tid][p] = max(out[tid].get(p, 0), g + a)
    return out


def odds(league_id, season, date_str, ttl=2 * 3600):
    """{fixture_id: {market: price}} using the first bookmaker that returns each market."""
    res = {}
    page = 1
    while True:
        d = _get("/odds", {"league": league_id, "season": season, "date": date_str, "page": page}, ttl)
        for x in d["response"]:
            m = {}
            for bk in x["bookmakers"]:
                for bet in bk["bets"]:
                    vals = {v["value"]: float(v["odd"]) for v in bet["values"]}
                    if bet["name"] == "Match Winner":
                        m.setdefault("home", vals.get("Home")); m.setdefault("draw", vals.get("Draw"))
                        m.setdefault("away", vals.get("Away"))
                    elif bet["name"] == "Goals Over/Under":
                        for line in ("1.5", "2.5", "3.5"):
                            if f"Over {line}" in vals:
                                m.setdefault(f"over{line}", vals[f"Over {line}"])
                                m.setdefault(f"under{line}", vals.get(f"Under {line}"))
                    elif bet["name"] == "Both Teams Score":
                        m.setdefault("btts_yes", vals.get("Yes")); m.setdefault("btts_no", vals.get("No"))
                if m:
                    m["bookmaker"] = bk["name"]
            res[x["fixture"]["id"]] = {k: v for k, v in m.items() if v}
        paging = d.get("paging", {})
        if page >= paging.get("total", 1):
            break
        page += 1
    return res
