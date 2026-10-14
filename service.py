"""Builds the daily prediction set."""
import json
from collections import defaultdict
from datetime import date, datetime

import numpy as np

from . import source
from .config import (LEAGUES, PRED_DIR, HALF_LIFE_DAYS, FORM_MATCHES, H2H_MATCHES, ML_WEIGHT,
                     HISTORY_SEASONS_BACK, current_season, DEMO, PROVIDER)
from .model import StrengthModel, score_matrix, rescale_1x2, markets, to_py
from .ml import MLModel
from . import context as C
from . import store


def load_history(today):
    season = current_season(today)
    out, seen = [], set()
    for lid in LEAGUES:
        for s in range(season - HISTORY_SEASONS_BACK, season + 1):
            try:
                for m in source.finished_matches(lid, s):
                    if m["id"] not in seen and m["hg"] is not None and date.fromisoformat(m["date"]) < today:
                        seen.add(m["id"]); out.append(m)
            except Exception as e:  # keep going if one league/season fails
                print(f"[warn] history {lid}/{s}: {e}")
    return sorted(out, key=lambda m: m["date"])


def fit_models(history, today):
    sm = StrengthModel(half_life_days=HALF_LIFE_DAYS).fit(history, today)
    ml = MLModel().fit(history)
    return sm, ml


def _kelly_free_value(p, odd):
    return p * odd - 1  # expected value per unit staked


def value_bets(M, o):
    pairs = [("1X2 Home", M["1X2"]["Home"], o.get("home")), ("1X2 Draw", M["1X2"]["Draw"], o.get("draw")),
             ("1X2 Away", M["1X2"]["Away"], o.get("away")),
             ("Over 1.5", M["Total goals over/under"]["Over 1.5"], o.get("over1.5")),
             ("Under 1.5", M["Total goals over/under"]["Under 1.5"], o.get("under1.5")),
             ("Over 2.5", M["Total goals over/under"]["Over 2.5"], o.get("over2.5")),
             ("Under 2.5", M["Total goals over/under"]["Under 2.5"], o.get("under2.5")),
             ("Over 3.5", M["Total goals over/under"]["Over 3.5"], o.get("over3.5")),
             ("Under 3.5", M["Total goals over/under"]["Under 3.5"], o.get("under3.5")),
             ("BTTS Yes", M["Both teams to score"]["Yes"], o.get("btts_yes")),
             ("BTTS No", M["Both teams to score"]["No"], o.get("btts_no"))]
    out = []
    for name, p, odd in pairs:
        if odd:
            out.append({"bet": name, "model_prob": round(p, 4), "odds": odd,
                        "implied_prob": round(1 / odd, 4), "edge": round(p - 1 / odd, 4),
                        "ev_per_unit": round(_kelly_free_value(p, odd), 4)})
    return sorted(out, key=lambda x: -x["ev_per_unit"])


def _notice():
    if PROVIDER == "footballdata" and not DEMO:
        return ("Free football-data.org data: Premier League, La Liga, Serie A, Primeira Liga and Champions "
                "League only. Injuries and bookmaker odds are not available, so those adjustments are off.")
    return None


def build(day: date, history=None, models=None):
    ds = day.isoformat()
    fixtures = source.fixtures_on(ds)
    now = datetime.utcnow().isoformat()
    if not fixtures:
        out = {"date": ds, "matches": [], "generated_at": now, "status_updated_at": now,
               "demo": DEMO, "notice": _notice()}
        (PRED_DIR / f"{ds}.json").write_text(json.dumps(out))
        return out
    history = history if history is not None else load_history(day)
    if len(history) < 100:
        raise RuntimeError("Not enough match history loaded (%d matches). Check the API token/plan "
                           "and the server logs for [warn] lines." % len(history))
    sm, ml = models if models else fit_models(history, day)
    season = current_season(day)

    lids = sorted({f["league_id"] for f in fixtures})
    inj, contrib, odds_by_fx = {}, defaultdict(dict), {}
    for lid in lids:
        try: inj.update(source.injuries(lid, season, ds))
        except Exception as e: print(f"[warn] injuries {lid}: {e}")
        try: contrib.update(source.top_contributors(lid, season))
        except Exception as e: print(f"[warn] contributors {lid}: {e}")
        try: odds_by_fx.update(source.odds(lid, season, ds))
        except Exception as e: print(f"[warn] odds {lid}: {e}")

    # season goals per team (denominator for a player's share of team output)
    team_goals = defaultdict(int)
    for m in history:
        if date.fromisoformat(m["date"]).year >= day.year - 1:
            team_goals[m["home_id"]] += m["hg"]; team_goals[m["away_id"]] += m["ag"]

    results = []
    for f in fixtures:
        h, a = f["home_id"], f["away_id"]
        lh0, la0 = sm.lambdas(h, a)
        fh = C.form_summary(history, h, day, FORM_MATCHES, sm)
        fa = C.form_summary(history, a, day, FORM_MATCHES, sm)
        h2h = C.head_to_head(history, h, a, day, H2H_MATCHES)
        mf_h, mf_a = C.form_factors(fh, fa)
        mh_h, mh_a = C.h2h_factor(h2h)
        mi_h, mi_a, nh, na = C.injury_factors(inj.get(h, []), inj.get(a, []), contrib.get(h, {}), contrib.get(a, {}),
                                              team_goals[h] or 40, team_goals[a] or 40)
        lh = max(0.15, lh0 * mf_h * mh_h * mi_h)
        la = max(0.15, la0 * mf_a * mh_a * mi_a)

        P = score_matrix(lh, la, sm.rho)
        i, j = np.indices(P.shape)
        stat = tuple(float(x) for x in (P[i > j].sum(), P[i == j].sum(), P[i < j].sum()))
        mlp = ml.predict(h, a)
        if mlp:
            blend = [(1 - ML_WEIGHT) * s + ML_WEIGHT * m for s, m in zip(stat, mlp)]
            P = rescale_1x2(P, blend)
        M = markets(P, lh, la)
        final_1x2 = M["1X2"]
        top = max(final_1x2, key=final_1x2.get)
        sorted_p = sorted(final_1x2.values(), reverse=True)

        o = odds_by_fx.get(f["id"], {})
        vb = value_bets(M, o) if o else []
        best_ev = [v for v in vb if v["ev_per_unit"] > 0.05 and v["edge"] > 0.03][:3]

        results.append({
            "fixture_id": f["id"], "kickoff": f["kickoff"], "league_id": f["league_id"], "league": f["league"],
            "home": f["home"], "away": f["away"],
            "status": status_group(f["status"]), "score": _score(f),
            "home_crest": f.get("home_crest"), "away_crest": f.get("away_crest"), "league_emblem": f.get("league_emblem"),
            "expected_goals": {"home": round(lh, 2), "away": round(la, 2)},
            "pick": {"market": "1X2", "selection": top, "probability": round(float(final_1x2[top]), 4),
                     "confidence": "High" if sorted_p[0] > 0.60 else "Medium" if sorted_p[0] > 0.47 else "Low"},
            "markets": to_py(M),
            "factors": {
                "base_expected_goals": {"home": round(lh0, 2), "away": round(la0, 2)},
                "form_multiplier": {"home": round(mf_h, 3), "away": round(mf_a, 3)},
                "h2h_multiplier": {"home": round(mh_h, 3), "away": round(mh_a, 3)},
                "injury_multiplier": {"home": round(mi_h, 3), "away": round(mi_a, 3)},
                "stat_1x2": [round(x, 3) for x in stat],
                "ml_1x2": [round(x, 3) for x in mlp] if mlp else None,
                "dixon_coles_rho": round(sm.rho, 3),
            },
            "form": {"home": {k: v for k, v in fh.items() if k not in ("attack_ratio", "defence_ratio")},
                     "away": {k: v for k, v in fa.items() if k not in ("attack_ratio", "defence_ratio")}},
            "home_away_split": {"home_team_at_home": C.split_form(history, h, day, True),
                                "away_team_away": C.split_form(history, a, day, False)},
            "h2h": h2h,
            "elo": {"home": ml.elo(h), "away": ml.elo(a)},
            "injuries": {"home": inj.get(h, []), "away": inj.get(a, []),
                         "key_absences_home": nh, "key_absences_away": na},
            "odds": o, "value_bets": vb, "suggested_value": best_ev,
            "data_quality": "ok" if sm.known(h) and sm.known(a) else "limited (team has little history)",
        })
    results.sort(key=lambda r: r["kickoff"])
    now = datetime.utcnow().isoformat()
    out = {"date": ds, "generated_at": now, "status_updated_at": now, "demo": DEMO, "notice": _notice(),
           "matches": results, "n_history_matches": len(history)}
    (PRED_DIR / f"{ds}.json").write_text(json.dumps(out))
    try:  # permanent record for the Accuracy page (pre-kick-off predictions only)
        upcoming = [m for m in results if m["status"] == "upcoming"]
        store.log_predictions(ds, results, pick_surest(upcoming, 2))
    except Exception as e:
        print("[warn] could not log predictions:", e)
    return out


def get_cached(ds: str):
    f = PRED_DIR / f"{ds}.json"
    return json.loads(f.read_text()) if f.exists() else None


# ------------------------------- live status / scores ----------------------------------
def age_seconds(ts):
    try:
        return (datetime.utcnow() - datetime.fromisoformat(ts)).total_seconds()
    except Exception:
        return 1e12


def status_group(raw):
    raw = (raw or "").upper()
    if raw in ("FINISHED", "FT", "AET", "PEN"):
        return "finished"
    if raw in ("IN_PLAY", "PAUSED", "LIVE", "EXTRA_TIME", "PENALTY_SHOOTOUT", "1H", "2H", "HT", "ET", "P"):
        return "live"
    return "upcoming"


def _score(f):
    if f.get("hg") is None or f.get("ag") is None:
        return None
    return {"home": f["hg"], "away": f["ag"]}


def refresh_statuses(data):
    """Cheap update of a cached day: new kick-off status and scores only (no model re-fit)."""
    ds = data["date"]
    data["status_updated_at"] = datetime.utcnow().isoformat()  # set first so a failure is not retried at once
    try:
        by_id = {f["id"]: f for f in source.fixtures_on(ds)}
        for m in data["matches"]:
            f = by_id.get(m["fixture_id"])
            if f:
                m["status"] = status_group(f["status"])
                m["score"] = _score(f)
                m["kickoff"] = f["kickoff"]
    finally:
        (PRED_DIR / f"{ds}.json").write_text(json.dumps(data))
    return data


# ------------------------------- surest picks ------------------------------------------
# Only everyday markets are eligible. Over 0.5 goals, Over 5.5 etc. are near-certain but pay almost nothing,
# and "draw no bet" can return the stake, so they are left out.
_OU_LINES = ("1.5", "2.5", "3.5")


def _candidates(m):
    M, h, a = m["markets"], m["home"], m["away"]
    out = []
    for key, label in (("Home", f"{h} to win"), ("Draw", "Draw"), ("Away", f"{a} to win")):
        out.append(("1X2", key, label, M["1X2"][key]))
    for key, label in (("1X", f"{h} or draw (1X)"), ("X2", f"{a} or draw (X2)"), ("12", "No draw (12)")):
        out.append(("Double chance", key, label, M["Double chance"][key]))
    for line in _OU_LINES:
        for side in ("Over", "Under"):
            k = f"{side} {line}"
            out.append(("Goals", k, f"{side} {line} goals", M["Total goals over/under"][k]))
    for key in ("Yes", "No"):
        out.append(("BTTS", key, f"Both teams to score: {key}", M["Both teams to score"][key]))
    return out


def evaluate(market, key, hg, ag):
    """True/False once the score is known (None otherwise)."""
    if hg is None or ag is None:
        return None
    if market == "1X2":
        return {"Home": hg > ag, "Draw": hg == ag, "Away": hg < ag}[key]
    if market == "Double chance":
        return {"1X": hg >= ag, "X2": ag >= hg, "12": hg != ag}[key]
    if market == "Goals":
        side, line = key.split()
        return (hg + ag > float(line)) if side == "Over" else (hg + ag < float(line))
    if market == "BTTS":
        both = hg > 0 and ag > 0
        return both if key == "Yes" else not both
    return None


def _confidence(p):
    return "Very high" if p >= 0.80 else "High" if p >= 0.70 else "Medium" if p >= 0.60 else "Low"


def _reasons(m):
    r = [f"Expected goals: {m['home']} {m['expected_goals']['home']} - {m['expected_goals']['away']} {m['away']}"]
    fh, fa = m["form"]["home"], m["form"]["away"]
    if fh.get("n") and fa.get("n"):
        r.append(f"Last {min(fh['n'], fa['n'])} games: {m['home']} {fh['string']} ({fh['ppg']} pts/game), "
                 f"{m['away']} {fa['string']} ({fa['ppg']} pts/game)")
    sp = m.get("home_away_split", {})
    ha, aw = sp.get("home_team_at_home", {}), sp.get("away_team_away", {})
    if ha.get("n") and aw.get("n"):
        r.append(f"{m['home']} at home: {ha['ppg']} pts/game; {m['away']} away: {aw['ppg']} pts/game")
    h2h = m.get("h2h", {})
    if h2h.get("n", 0) >= 3:
        r.append(f"Head to head (last {h2h['n']}): {h2h['home_wins']}W-{h2h['draws']}D-{h2h['away_wins']}L "
                 f"for {m['home']}")
    return r


def _pick_dict(m, market, key, label, p):
    sc = m.get("score") or {}
    ok = evaluate(market, key, sc.get("home"), sc.get("away")) if m.get("status") == "finished" else None
    return {
        "fixture_id": m["fixture_id"], "league": m["league"], "kickoff": m["kickoff"],
        "home": m["home"], "away": m["away"], "status": m.get("status", "upcoming"), "score": m.get("score"),
        "home_crest": m.get("home_crest"), "away_crest": m.get("away_crest"), "league_emblem": m.get("league_emblem"),
        "market": market, "key": key, "bet": label, "probability": round(p, 4),
        "fair_odds": round(1 / p, 2) if p > 0 else None, "confidence": _confidence(p),
        "reasons": _reasons(m), "result": None if ok is None else ("won" if ok else "lost"),
    }


def pick_from_logged(m, logged):
    """Rebuild a surest pick for display from the one that was saved before kick-off."""
    return _pick_dict(m, logged["market"], logged["key"], logged["bet"], logged["probability"])


def pick_surest(matches, n=2):
    """The n games where the model's single most probable everyday bet is the most probable of the day."""
    best = []
    for m in matches:
        if m.get("data_quality") != "ok":
            continue
        if m["form"]["home"].get("n", 0) < 5 or m["form"]["away"].get("n", 0) < 5:
            continue
        cands = [c for c in _candidates(m) if c[3] < 0.95]
        if not cands:
            continue
        market, key, label, p = max(cands, key=lambda c: c[3])
        best.append(_pick_dict(m, market, key, label, p))
    best.sort(key=lambda x: -x["probability"])
    return best[:n]


def slip_options(m):
    """Every bet a user may put on a slip for this game, with the model's probability."""
    return [{"market": mk, "key": k, "label": lab, "probability": round(p, 4)} for mk, k, lab, p in _candidates(m)]


def settle_all():
    """Look up final scores for finished games and settle the accuracy log and everyone's slips."""
    pend = store.unsettled() + store.slip_pending()
    if not pend:
        return 0
    ids = sorted({p[0] for p in pend})
    dates = sorted({p[1] for p in pend})
    res = source.results_for(ids, dates)
    n = store.settle(res)
    n += store.slip_apply(res)
    return n


def table_for(league_id):
    return source.standings(league_id)
