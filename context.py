"""Match context: recent form, head-to-head, home strength, injuries -> goal-rate adjustments."""
from datetime import date
import numpy as np


def team_matches(matches, team_id, before: date):
    ms = [m for m in matches if team_id in (m["home_id"], m["away_id"]) and date.fromisoformat(m["date"]) < before]
    return sorted(ms, key=lambda m: m["date"])


def _result(m, team_id):
    home = m["home_id"] == team_id
    gf, ga = (m["hg"], m["ag"]) if home else (m["ag"], m["hg"])
    r = "W" if gf > ga else "D" if gf == ga else "L"
    return r, gf, ga, home


def form_summary(matches, team_id, before, n=8, model=None):
    last = team_matches(matches, team_id, before)[-n:]
    if not last:
        return {"string": "", "points": 0, "ppg": None, "gf": 0, "ga": 0, "n": 0,
                "attack_ratio": 1.0, "defence_ratio": 1.0}
    seq, pts, gf, ga, exp_gf, exp_ga = [], 0, 0, 0, 0.0, 0.0
    for m in last:
        r, f, a, home = _result(m, team_id)
        seq.append(r); pts += {"W": 3, "D": 1, "L": 0}[r]; gf += f; ga += a
        if model is not None and model.known(m["home_id"]) and model.known(m["away_id"]):
            lh, la = model.lambdas(m["home_id"], m["away_id"])
            e_f, e_a = (lh, la) if home else (la, lh)
        else:
            e_f = e_a = 1.35
        exp_gf += e_f; exp_ga += e_a
    return {"string": "".join(seq), "points": pts, "ppg": round(pts / len(last), 2), "gf": gf, "ga": ga, "n": len(last),
            "attack_ratio": (gf + 1) / (exp_gf + 1), "defence_ratio": (ga + 1) / (exp_ga + 1)}


def split_form(matches, team_id, before, home: bool, n=8):
    ms = [m for m in team_matches(matches, team_id, before) if (m["home_id"] == team_id) == home][-n:]
    if not ms:
        return {"n": 0}
    pts = gf = ga = 0
    for m in ms:
        r, f, a, _ = _result(m, team_id)
        pts += {"W": 3, "D": 1, "L": 0}[r]; gf += f; ga += a
    k = len(ms)
    return {"n": k, "ppg": round(pts / k, 2), "gf_pg": round(gf / k, 2), "ga_pg": round(ga / k, 2)}


def head_to_head(matches, home_id, away_id, before, n=10):
    ms = [m for m in matches if {m["home_id"], m["away_id"]} == {home_id, away_id}
          and date.fromisoformat(m["date"]) < before]
    ms = sorted(ms, key=lambda m: m["date"])[-n:]
    hw = dw = aw = 0
    gd = 0.0
    for m in ms:
        r, f, a, _ = _result(m, home_id)
        hw += r == "W"; dw += r == "D"; aw += r == "L"; gd += f - a
    k = len(ms)
    return {"n": k, "home_wins": hw, "draws": dw, "away_wins": aw,
            "avg_goal_diff_for_home": round(gd / k, 2) if k else 0.0,
            "recent": [{"date": m["date"], "score": f"{m['home']} {m['hg']}-{m['ag']} {m['away']}"} for m in ms[-5:]]}


def form_factors(fh, fa):
    """Shrunk multipliers (max about +-12%) to goal rates from recent over/under-performance."""
    def f(x): return float(np.clip(x, 0.80, 1.25) ** 0.25)
    # home attack vs away defence, away attack vs home defence
    fh_att, fh_def = f(fh["attack_ratio"]), f(fh["defence_ratio"])
    fa_att, fa_def = f(fa["attack_ratio"]), f(fa["defence_ratio"])
    w_h = min(fh["n"], 8) / 8; w_a = min(fa["n"], 8) / 8
    mult_h = 1 + w_h * (fh_att - 1) + w_a * (fa_def - 1)
    mult_a = 1 + w_a * (fa_att - 1) + w_h * (fh_def - 1)
    return mult_h, mult_a


def h2h_factor(h2h):
    if h2h["n"] < 3:
        return 1.0, 1.0
    w = min(h2h["n"], 10) / 10
    g = float(np.clip(h2h["avg_goal_diff_for_home"], -2, 2))
    return float(np.exp(0.03 * w * g)), float(np.exp(-0.03 * w * g))


def injury_factors(inj_home, inj_away, contrib_home, contrib_away, team_goals_home, team_goals_away):
    """Attack penalty for missing top scorers/creators; small generic penalty for other absentees
    (their defensive weight raises the opponent's rate)."""
    def one(injured, contrib, team_goals):
        share, named = 0.0, []
        for p in injured:
            pts = contrib.get(p["player_id"], 0)
            if pts:
                s = min(pts / max(team_goals, 20), 0.35)
                share += s
                named.append({"name": p["name"], "contribution": pts, "share": round(s, 3)})
        att = 1 - min(0.20, 0.45 * share)
        others = len([p for p in injured if p["player_id"] not in contrib])
        dfn = 1 + min(0.06, 0.012 * others + 0.5 * share * 0.2)   # opponent scores a bit more
        return att, dfn, named
    ah, dh, nh = one(inj_home, contrib_home, team_goals_home)
    aa, da, na = one(inj_away, contrib_away, team_goals_away)
    return ah * da, aa * dh, nh, na  # (home rate mult, away rate mult)
