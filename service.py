"""Builds the daily prediction set."""
import json
from collections import defaultdict
from datetime import date, datetime

import numpy as np

from . import source
from .config import (LEAGUES, PRED_DIR, HALF_LIFE_DAYS, FORM_MATCHES, H2H_MATCHES, ML_WEIGHT,
                     HISTORY_SEASONS_BACK, current_season, DEMO)
from .model import StrengthModel, score_matrix, rescale_1x2, markets, to_py
from .ml import MLModel
from . import context as C


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


def build(day: date, history=None, models=None):
    history = history if history is not None else load_history(day)
    sm, ml = models if models else fit_models(history, day)
    season = current_season(day)
    ds = day.isoformat()
    fixtures = source.fixtures_on(ds)
    if not fixtures:
        return {"date": ds, "matches": [], "generated_at": datetime.utcnow().isoformat(), "demo": DEMO}

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
    out = {"date": ds, "generated_at": datetime.utcnow().isoformat(), "demo": DEMO,
           "matches": results, "n_history_matches": len(history)}
    (PRED_DIR / f"{ds}.json").write_text(json.dumps(out))
    return out


def get_cached(ds: str):
    f = PRED_DIR / f"{ds}.json"
    return json.loads(f.read_text()) if f.exists() else None
