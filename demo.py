"""Synthetic data source with the same interface as apifootball.py, so the app
runs end-to-end without an API key. Teams have hidden strengths; results are simulated."""
import random
from datetime import date, datetime, timedelta
import numpy as np

from .config import LEAGUES, current_season

_rng_seed = 42
_TEAMS = {}
_NAMES = ["Atlas", "Borough", "Capital", "Dynamo", "Eagles", "Forest", "Giants", "Harbor", "Union",
          "Jets", "Kings", "Lions", "Metro", "Norte", "Olympic", "Phoenix", "Quest", "Rovers",
          "Stars", "Titans"]


def _teams(league_id):
    if league_id not in _TEAMS:
        rng = random.Random(league_id * 7 + _rng_seed)
        pool = []
        for i, n in enumerate(_NAMES[: (36 if LEAGUES[league_id]["cup"] else 18)] if not LEAGUES[league_id]["cup"] else _NAMES):
            tid = league_id * 100 + i
            pool.append({"id": tid, "name": f"{LEAGUES[league_id]['country'][:3].upper()} {n}",
                         "att": rng.gauss(0, 0.28), "dfn": rng.gauss(0, 0.28), "home": rng.gauss(0.25, 0.08)})
        _TEAMS[league_id] = pool
    return _TEAMS[league_id]


def _sim(t_h, t_a, rng):
    lh = np.exp(0.15 + t_h["att"] - t_a["dfn"] + t_h["home"])
    la = np.exp(0.15 + t_a["att"] - t_h["dfn"])
    return rng.poisson(lh), rng.poisson(la), lh, la


def finished_matches(league_id, season, ttl=0):
    rng = np.random.default_rng(league_id + season)
    teams = _teams(league_id)
    start = date(season, 8, 10)
    out, mid = [], league_id * 100000 + season * 10
    n_rounds = 14 if LEAGUES[league_id]["cup"] else 30
    for r in range(n_rounds):
        d = start + timedelta(days=7 * r)
        if d >= date.today():
            break
        order = list(range(len(teams)))
        rng.shuffle(order)
        for k in range(0, len(order) - 1, 2):
            h, a = teams[order[k]], teams[order[k + 1]]
            hg, ag, _, _ = _sim(h, a, rng)
            mid += 1
            out.append({"id": mid, "date": d.isoformat(), "kickoff": d.isoformat() + "T19:00:00", "status": "FT",
                        "league_id": league_id, "league": LEAGUES[league_id]["name"],
                        "home_id": h["id"], "home": h["name"], "away_id": a["id"], "away": a["name"],
                        "hg": int(hg), "ag": int(ag),
                        "ht_hg": int(rng.binomial(hg, 0.45)), "ht_ag": int(rng.binomial(ag, 0.45))})
    return out


def fixtures_on(date_str, ttl=0):
    rng = random.Random(date_str)
    out = []
    for lid in LEAGUES:
        teams = _teams(lid)[:]
        rng.shuffle(teams)
        for k in range(0, min(len(teams), 8 if lid not in (2, 3, 848) else 6) - 1, 2):
            h, a = teams[k], teams[k + 1]
            hh = rng.choice([17, 18, 19, 20, 21])
            out.append({"id": int(date_str.replace("-", "")) * 1000 + lid + k, "date": date_str,
                        "kickoff": f"{date_str}T{hh}:00:00+01:00", "status": "NS", "league_id": lid,
                        "league": LEAGUES[lid]["name"], "home_id": h["id"], "home": h["name"],
                        "away_id": a["id"], "away": a["name"], "hg": None, "ag": None, "ht_hg": None, "ht_ag": None})
    return out


def injuries(league_id, season, date_str, ttl=0):
    rng = random.Random(f"{league_id}{date_str}")
    res = {}
    for t in _teams(league_id):
        res[t["id"]] = [{"player_id": t["id"] * 100 + j, "name": f"Player {t['id']}-{j}", "type": "Missing Fixture",
                         "reason": rng.choice(["Knee Injury", "Hamstring", "Suspended", "Illness"])}
                        for j in range(rng.choice([0, 0, 1, 1, 2, 3]))]
    return res


def top_contributors(league_id, season, ttl=0):
    rng = random.Random(league_id)
    return {t["id"]: {t["id"] * 100 + j: rng.randint(2, 14) for j in range(3)} for t in _teams(league_id)}


def odds(league_id, season, date_str, ttl=0):
    return {}  # demo has no bookmaker; value-bet section is skipped


def results_for(ids, dates=(), ttl=0):
    return {}  # demo games never finish


def standings(league_id, ttl=0):
    rows = {}
    for t in _teams(league_id):
        rows[t["id"]] = {"team": t["name"], "crest": None, "played": 0, "won": 0, "draw": 0, "lost": 0, "gf": 0, "ga": 0}
    for m in finished_matches(league_id, current_season(date.today())):
        h, a = rows[m["home_id"]], rows[m["away_id"]]
        for r, f, c in ((h, m["hg"], m["ag"]), (a, m["ag"], m["hg"])):
            r["played"] += 1; r["gf"] += f; r["ga"] += c
            r["won"] += f > c; r["draw"] += f == c; r["lost"] += f < c
    table = sorted(rows.values(), key=lambda r: (-(3 * r["won"] + r["draw"]), -(r["gf"] - r["ga"]), -r["gf"]))
    for i, r in enumerate(table, 1):
        r.update(pos=i, points=3 * r["won"] + r["draw"], gd=r["gf"] - r["ga"])
    scorers = [{"player": f"Player {i + 1}", "team": table[i % len(table)]["team"], "crest": None,
                "goals": 12 - i, "assists": i % 4, "penalties": i % 2, "played": 9} for i in range(8)]
    return {"table": table, "scorers": scorers, "emblem": None}
