"""football-data.org (v4) client. Free tier: current season, 10 requests/minute,
competitions PL, PD, SA, PPL, CL. No injuries or odds on the free tier."""
import hashlib
import json
import threading
import time
from datetime import date, datetime, timedelta

import requests

from .config import FD_TOKEN, CACHE_DIR, LEAGUES, tzinfo

BASE = "https://api.football-data.org/v4"

# football-data competition code -> our league id (see config.LEAGUES)
CODE_TO_LID = {"PL": 39, "PD": 140, "SA": 135, "PPL": 94, "CL": 2}
LID_TO_CODE = {v: k for k, v in CODE_TO_LID.items()}
SUPPORTED = set(CODE_TO_LID.values())

_calls = []  # timestamps of recent requests (sliding 60s window, limit 10/min)
_tlock = threading.Lock()


def _throttle():
    with _tlock:
        now = time.time()
        while _calls and now - _calls[0] > 60:
            _calls.pop(0)
        if len(_calls) >= 9:
            wait = 61 - (now - _calls[0])
            if wait > 0:
                time.sleep(wait)
            now = time.time()
            while _calls and now - _calls[0] > 60:
                _calls.pop(0)
        _calls.append(time.time())


def _get(path, params, ttl):
    key = hashlib.md5(("fd" + path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    f = CACHE_DIR / f"{key}.json"
    if f.exists() and time.time() - f.stat().st_mtime < ttl:
        return json.loads(f.read_text())
    for attempt in range(2):
        _throttle()
        r = requests.get(BASE + path, params=params, headers={"X-Auth-Token": FD_TOKEN}, timeout=30)
        if r.status_code == 429 and attempt == 0:
            time.sleep(62)
            continue
        break
    if r.status_code != 200:
        raise RuntimeError(f"football-data.org {r.status_code}: {r.text[:200]}")
    data = r.json()
    f.write_text(json.dumps(data))
    return data


def _ft(x):
    sc = x.get("score", {}) or {}
    return sc.get("regularTime") or sc.get("fullTime") or {}   # 90-minute score where available


def _norm(x, lid, emblem=None):
    ft = _ft(x)
    ht = (x.get("score", {}) or {}).get("halfTime") or {}
    return {
        "id": x["id"],
        "date": x["utcDate"][:10],
        "kickoff": x["utcDate"],
        "status": x["status"],
        "league_id": lid,
        "league": LEAGUES[lid]["name"],
        "league_emblem": emblem,
        "home_id": x["homeTeam"]["id"], "home": x["homeTeam"].get("shortName") or x["homeTeam"]["name"],
        "away_id": x["awayTeam"]["id"], "away": x["awayTeam"].get("shortName") or x["awayTeam"]["name"],
        "home_crest": x["homeTeam"].get("crest"), "away_crest": x["awayTeam"].get("crest"),
        "hg": ft.get("home"), "ag": ft.get("away"),
        "ht_hg": ht.get("home"), "ht_ag": ht.get("away"),
    }


def finished_matches(league_id, season, ttl=6 * 3600):
    if league_id not in SUPPORTED:
        return []
    code = LID_TO_CODE[league_id]
    d = _get(f"/competitions/{code}/matches", {"season": season, "status": "FINISHED"}, ttl)
    emb = (d.get("competition") or {}).get("emblem")
    out = [_norm(x, league_id, emb) for x in d.get("matches", [])]
    return [m for m in out if m["hg"] is not None and m["ag"] is not None]


SHOWN = ("SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "LIVE", "EXTRA_TIME", "PENALTY_SHOOTOUT", "FINISHED")


def _local_date(utc_str):
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    return dt.astimezone(tzinfo()).date().isoformat()


def _window(date_str, ttl):
    """Raw matches around a date (the same request is shared by fixtures and results, so it is cached once)."""
    day = date.fromisoformat(date_str)
    lo, hi = (day - timedelta(days=1)).isoformat(), (day + timedelta(days=1)).isoformat()
    out = []
    for lid, code in LID_TO_CODE.items():
        try:
            d = _get(f"/competitions/{code}/matches", {"dateFrom": lo, "dateTo": hi}, ttl)
        except Exception as e:
            print(f"[warn] fixtures {code}: {e}")
            continue
        emb = (d.get("competition") or {}).get("emblem")
        out.extend((lid, x, emb) for x in d.get("matches", []))
    return out


def fixtures_on(date_str, ttl=600):
    """All games whose kick-off falls on `date_str` in the app's time zone (upcoming, live and finished)."""
    return [_norm(x, lid, emb) for lid, x, emb in _window(date_str, ttl)
            if x["status"] in SHOWN and _local_date(x["utcDate"]) == date_str]


def results_on(date_str, ttl=600):
    out = {}
    for lid, x, emb in _window(date_str, ttl):
        ft = _ft(x)
        if x["status"] == "FINISHED" and ft.get("home") is not None and ft.get("away") is not None:
            out[x["id"]] = (ft["home"], ft["away"])
    return out


def results_for(ids, dates=(), ttl=300):
    """{match_id: (home_goals, away_goals)} for finished games. One request for many ids when possible,
    falling back to day-by-day lookups."""
    ids = [int(i) for i in ids]
    out = {}
    if not ids:
        return out
    try:
        for i in range(0, len(ids), 40):
            d = _get("/matches", {"ids": ",".join(str(j) for j in ids[i:i + 40])}, ttl)
            for x in d.get("matches", []):
                ft = _ft(x)
                if x["status"] == "FINISHED" and ft.get("home") is not None and ft.get("away") is not None:
                    out[x["id"]] = (ft["home"], ft["away"])
        return out
    except Exception as e:
        print(f"[warn] results by id failed ({e}); trying by date")
    for ds in dates:
        try:
            out.update(results_on(ds, ttl))
        except Exception as e:
            print(f"[warn] results {ds}: {e}")
    return out


def standings(league_id, ttl=3600):
    code = LID_TO_CODE.get(league_id)
    if not code:
        return {"table": [], "scorers": [], "emblem": None}
    d = _get(f"/competitions/{code}/standings", {}, ttl)
    tables = d.get("standings", [])
    pick = [t for t in tables if t.get("type") == "TOTAL"] or tables
    table = []
    for r in (pick[0].get("table", []) if pick else []):
        t = r.get("team", {})
        table.append({"pos": r.get("position"), "team": t.get("shortName") or t.get("name"), "crest": t.get("crest"),
                      "played": r.get("playedGames"), "won": r.get("won"), "draw": r.get("draw"), "lost": r.get("lost"),
                      "gf": r.get("goalsFor"), "ga": r.get("goalsAgainst"), "gd": r.get("goalDifference"),
                      "points": r.get("points")})
    scorers = []
    try:
        s = _get(f"/competitions/{code}/scorers", {"limit": 10}, ttl)
        for x in s.get("scorers", []):
            t = x.get("team", {})
            scorers.append({"player": (x.get("player") or {}).get("name"), "team": t.get("shortName") or t.get("name"),
                            "crest": t.get("crest"), "goals": x.get("goals") or 0, "assists": x.get("assists") or 0,
                            "penalties": x.get("penalties") or 0, "played": x.get("playedMatches")})
    except Exception as e:
        print(f"[warn] scorers {code}: {e}")
    return {"table": table, "scorers": scorers, "emblem": (d.get("competition") or {}).get("emblem")}


# Not available on the free tier
def injuries(league_id, season, date_str, ttl=0):
    return {}


def top_contributors(league_id, season, ttl=0):
    return {}


def odds(league_id, season, date_str, ttl=0):
    return {}
