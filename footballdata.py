"""football-data.org (v4) client. Free tier: current season, 10 requests/minute,
competitions PL, PD, SA, PPL, CL. No injuries or odds on the free tier."""
import hashlib
import json
import time
from datetime import date, timedelta

import requests

from .config import FD_TOKEN, CACHE_DIR, LEAGUES

BASE = "https://api.football-data.org/v4"

# football-data competition code -> our league id (see config.LEAGUES)
CODE_TO_LID = {"PL": 39, "PD": 140, "SA": 135, "PPL": 94, "CL": 2}
LID_TO_CODE = {v: k for k, v in CODE_TO_LID.items()}
SUPPORTED = set(CODE_TO_LID.values())

_calls = []  # timestamps of recent requests (sliding 60s window, limit 10/min)


def _throttle():
    now = time.time()
    while _calls and now - _calls[0] > 60:
        _calls.pop(0)
    if len(_calls) >= 9:
        wait = 61 - (now - _calls[0])
        if wait > 0:
            time.sleep(wait)
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


def _norm(x, lid):
    sc = x.get("score", {}) or {}
    ft = sc.get("regularTime") or sc.get("fullTime") or {}   # 90-minute score where available
    ht = sc.get("halfTime") or {}
    return {
        "id": x["id"],
        "date": x["utcDate"][:10],
        "kickoff": x["utcDate"],
        "status": x["status"],
        "league_id": lid,
        "league": LEAGUES[lid]["name"],
        "home_id": x["homeTeam"]["id"], "home": x["homeTeam"].get("shortName") or x["homeTeam"]["name"],
        "away_id": x["awayTeam"]["id"], "away": x["awayTeam"].get("shortName") or x["awayTeam"]["name"],
        "hg": ft.get("home"), "ag": ft.get("away"),
        "ht_hg": ht.get("home"), "ht_ag": ht.get("away"),
    }


def finished_matches(league_id, season, ttl=6 * 3600):
    if league_id not in SUPPORTED:
        return []
    code = LID_TO_CODE[league_id]
    d = _get(f"/competitions/{code}/matches", {"season": season, "status": "FINISHED"}, ttl)
    out = [_norm(x, league_id) for x in d.get("matches", [])]
    return [m for m in out if m["hg"] is not None and m["ag"] is not None]


def fixtures_on(date_str, ttl=1800):
    day = date.fromisoformat(date_str)
    nxt = (day + timedelta(days=1)).isoformat()
    out = []
    for lid, code in LID_TO_CODE.items():
        try:
            d = _get(f"/competitions/{code}/matches",
                     {"dateFrom": date_str, "dateTo": nxt}, ttl)
        except Exception as e:
            print(f"[warn] fixtures {code}: {e}")
            continue
        for x in d.get("matches", []):
            if x["status"] in ("SCHEDULED", "TIMED") and x["utcDate"][:10] == date_str:
                out.append(_norm(x, lid))
    return out


# Not available on the free tier
def injuries(league_id, season, date_str, ttl=0):
    return {}


def top_contributors(league_id, season, ttl=0):
    return {}


def odds(league_id, season, date_str, ttl=0):
    return {}
