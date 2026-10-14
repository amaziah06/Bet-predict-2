"""Saved data: the permanent log of predictions (for the Accuracy page) and each user's bet slip."""
import time
from datetime import datetime, timedelta, timezone

from . import db


def _kick(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _past(ts, hours):
    k = _kick(ts)
    if k is None:
        return False
    if k.tzinfo is None:
        k = k.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - k > timedelta(hours=hours)


def _outcome(hg, ag):
    return "Home" if hg > ag else "Draw" if hg == ag else "Away"


# ------------------------------------ prediction log -------------------------------------
def log_predictions(ds, matches, surest):
    """Save each reliable, not-yet-started prediction ONCE (first save wins, so results can never be
    'improved' afterwards). The day's surest picks are fixed the same way."""
    rows = [m for m in matches if m.get("status") == "upcoming" and m.get("data_quality") == "ok"]
    if not rows:
        return 0
    n = 0
    with db.connect() as c:
        for m in rows:
            H = m["markets"]["1X2"]
            cur = db.run(
                c, "INSERT INTO pred_log (fixture_id, match_date, league, home, away, kickoff, pick, pick_prob, "
                   "p_home, p_draw, p_away, logged_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                   "ON CONFLICT (fixture_id) DO NOTHING",
                (m["fixture_id"], ds, m["league"], m["home"], m["away"], m["kickoff"], m["pick"]["selection"],
                 m["pick"]["probability"], H["Home"], H["Draw"], H["Away"], int(time.time())))
            n += max(cur.rowcount, 0)
        if surest and not db.run(c, "SELECT 1 FROM pred_log WHERE match_date = ? AND surest = 1", (ds,)).fetchone():
            for p in surest:
                db.run(c, "UPDATE pred_log SET surest = 1, s_market = ?, s_key = ?, s_label = ?, s_prob = ? "
                          "WHERE fixture_id = ?", (p["market"], p["key"], p["bet"], p["probability"], p["fixture_id"]))
    return n


def logged_surest(ds):
    with db.connect() as c:
        rows = db.run(c, "SELECT fixture_id, s_market, s_key, s_label, s_prob FROM pred_log "
                         "WHERE match_date = ? AND surest = 1 ORDER BY s_prob DESC", (ds,)).fetchall()
    return [{"fixture_id": r[0], "market": r[1], "key": r[2], "bet": r[3], "probability": r[4]} for r in rows]


def unsettled(days=10):
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    with db.connect() as c:
        rows = db.run(c, "SELECT fixture_id, match_date, kickoff FROM pred_log WHERE settled = 0 AND match_date >= ?",
                      (since,)).fetchall()
    return [(r[0], r[1]) for r in rows if _past(r[2], 2.5)]


def settle(results):
    if not results:
        return 0
    n = 0
    with db.connect() as c:
        for fid, (hg, ag) in results.items():
            cur = db.run(c, "UPDATE pred_log SET hg = ?, ag = ?, settled = 1 WHERE fixture_id = ? AND settled = 0",
                         (hg, ag, int(fid)))
            n += max(cur.rowcount, 0)
    return n


def accuracy():
    from .service import evaluate
    with db.connect() as c:
        rows = db.run(c, "SELECT fixture_id, match_date, league, home, away, kickoff, pick, pick_prob, hg, ag, surest, "
                         "s_market, s_key, s_label, s_prob FROM pred_log WHERE settled = 1 "
                         "ORDER BY kickoff DESC").fetchall()
        first = db.run(c, "SELECT MIN(match_date), COUNT(*) FROM pred_log").fetchone()
        waiting = db.run(c, "SELECT COUNT(*) FROM pred_log WHERE settled = 0").fetchone()[0]
    items = []
    for r in rows:
        hit = _outcome(r[8], r[9]) == r[6]
        s_hit = evaluate(r[11], r[12], r[8], r[9]) if r[10] and r[11] else None
        items.append({"fixture_id": r[0], "date": r[1], "league": r[2], "home": r[3], "away": r[4], "pick": r[6],
                      "prob": r[7], "hg": r[8], "ag": r[9], "hit": hit, "surest": bool(r[10]),
                      "s_label": r[13], "s_prob": r[14], "s_hit": s_hit})

    def block(sub):
        n = len(sub)
        return {"n": n, "hits": sum(1 for x in sub if x["hit"]),
                "hit_rate": round(sum(1 for x in sub if x["hit"]) / n, 4) if n else None,
                "avg_confidence": round(sum(x["prob"] for x in sub) / n, 4) if n else None}

    sur = [x for x in items if x["surest"] and x["s_hit"] is not None]
    wins = sum(1 for x in sur if x["s_hit"])
    return {
        "since": first[0], "logged": first[1], "waiting": waiting,
        "overall": block(items),
        "bands": [{"label": "60% or more", **block([x for x in items if x["prob"] >= 0.60])},
                  {"label": "47% to 60%", **block([x for x in items if 0.47 <= x["prob"] < 0.60])},
                  {"label": "Under 47%", **block([x for x in items if x["prob"] < 0.47])}],
        "surest": {"n": len(sur), "wins": wins, "losses": len(sur) - wins,
                   "hit_rate": round(wins / len(sur), 4) if sur else None,
                   "avg_probability": round(sum(x["s_prob"] for x in sur) / len(sur), 4) if sur else None},
        "recent": items[:25],
    }


# ----------------------------------------- slips -----------------------------------------
SLIP_LIMIT = 25


def slip_add(user_id, match, opt):
    with db.connect() as c:
        if db.run(c, "SELECT 1 FROM slips WHERE user_id = ? AND fixture_id = ? AND market = ? AND bet_key = ?",
                  (user_id, match["fixture_id"], opt["market"], opt["key"])).fetchone():
            return "duplicate"
        if db.run(c, "SELECT COUNT(*) FROM slips WHERE user_id = ? AND status = 'pending'", (user_id,)).fetchone()[0] \
                >= SLIP_LIMIT:
            return "full"
        db.run(c, "INSERT INTO slips (user_id, fixture_id, match_date, league, home, away, kickoff, market, bet_key, "
                  "label, probability, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
               (user_id, match["fixture_id"], match["date"], match["league"], match["home"], match["away"],
                match["kickoff"], opt["market"], opt["key"], opt["label"], opt["probability"], int(time.time())))
    return "ok"


def slip_list(user_id):
    with db.connect() as c:
        rows = db.run(c, "SELECT id, fixture_id, match_date, league, home, away, kickoff, market, bet_key, label, "
                         "probability, status, hg, ag FROM slips WHERE user_id = ? ORDER BY created_at DESC, id DESC "
                         "LIMIT 100", (user_id,)).fetchall()
    return [{"id": r[0], "fixture_id": r[1], "date": r[2], "league": r[3], "home": r[4], "away": r[5], "kickoff": r[6],
             "market": r[7], "key": r[8], "label": r[9], "probability": r[10], "status": r[11], "hg": r[12],
             "ag": r[13]} for r in rows]


def slip_count(user_id):
    with db.connect() as c:
        return db.run(c, "SELECT COUNT(*) FROM slips WHERE user_id = ? AND status = 'pending'",
                      (user_id,)).fetchone()[0]


def slip_delete(user_id, item_id):
    with db.connect() as c:
        return db.run(c, "DELETE FROM slips WHERE id = ? AND user_id = ?", (item_id, user_id)).rowcount > 0


def slip_clear_settled(user_id):
    with db.connect() as c:
        return db.run(c, "DELETE FROM slips WHERE user_id = ? AND status <> 'pending'", (user_id,)).rowcount


def slip_pending(past_hours=2.5):
    """(fixture_id, date) of pending slip bets whose game should be over, across all users."""
    with db.connect() as c:
        rows = db.run(c, "SELECT DISTINCT fixture_id, match_date, kickoff FROM slips WHERE status = 'pending'").fetchall()
    return [(r[0], r[1]) for r in rows if _past(r[2], past_hours)]


def slip_apply(results):
    """Mark pending slip bets won/lost once the final score is known."""
    from .service import evaluate
    n = 0
    with db.connect() as c:
        for fid, (hg, ag) in results.items():
            rows = db.run(c, "SELECT id, market, bet_key FROM slips WHERE fixture_id = ? AND status = 'pending'",
                          (int(fid),)).fetchall()
            for sid, market, key in rows:
                ok = evaluate(market, key, hg, ag)
                if ok is None:
                    continue
                db.run(c, "UPDATE slips SET status = ?, hg = ?, ag = ? WHERE id = ?",
                       ("won" if ok else "lost", hg, ag, sid))
                n += 1
    return n
