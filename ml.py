"""Machine-learning layer: gradient-boosted 1X2 classifier on leak-free, walk-forward
features (Elo, last-8 form, home/away split form). It only nudges the statistical model."""
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

K = 24
HOME_ELO = 60


class _State:
    def __init__(self, n_form=8):
        self.elo = defaultdict(lambda: 1500.0)
        self.form = defaultdict(lambda: deque(maxlen=n_form))      # (pts, gf, ga)
        self.home_form = defaultdict(lambda: deque(maxlen=n_form))
        self.away_form = defaultdict(lambda: deque(maxlen=n_form))
        self.count = defaultdict(int)


def _agg(dq):
    if not dq:
        return 1.35, 1.35, 1.35
    a = np.array(dq, dtype=float)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()


def _feat(st, h, a):
    hp, hgf, hga = _agg(st.form[h]); ap, agf, aga = _agg(st.form[a])
    hhp, _, _ = _agg(st.home_form[h]); aap, _, _ = _agg(st.away_form[a])
    ed = st.elo[h] + HOME_ELO - st.elo[a]
    return [ed, hp, ap, hgf, hga, agf, aga, hhp, aap, hp - ap, (hgf - hga) - (agf - aga)]


def _update(st, m):
    h, a, hg, ag = m["home_id"], m["away_id"], m["hg"], m["ag"]
    exp_h = 1 / (1 + 10 ** (-(st.elo[h] + HOME_ELO - st.elo[a]) / 400))
    res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    mult = 1 + 0.5 * np.log1p(abs(hg - ag))
    st.elo[h] += K * mult * (res - exp_h); st.elo[a] -= K * mult * (res - exp_h)
    ph, pa = (3 if hg > ag else 1 if hg == ag else 0), (3 if ag > hg else 1 if hg == ag else 0)
    st.form[h].append((ph, hg, ag)); st.form[a].append((pa, ag, hg))
    st.home_form[h].append((ph, hg, ag)); st.away_form[a].append((pa, ag, hg))
    st.count[h] += 1; st.count[a] += 1


class MLModel:
    def __init__(self):
        self.clf = None
        self.state = None

    def fit(self, matches):
        ms = sorted(matches, key=lambda m: m["date"])
        st = _State()
        X, y = [], []
        for m in ms:
            if st.count[m["home_id"]] >= 5 and st.count[m["away_id"]] >= 5:
                X.append(_feat(st, m["home_id"], m["away_id"]))
                y.append(0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2)
            _update(st, m)
        self.state = st
        if len(X) < 400:
            return self  # not enough data; predict() returns None
        self.clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120,
                                                  l2_regularization=2.0, random_state=0)
        self.clf.fit(np.array(X), np.array(y))
        return self

    def predict(self, home_id, away_id):
        """-> (home, draw, away) or None"""
        if self.clf is None or self.state.count[home_id] < 5 or self.state.count[away_id] < 5:
            return None
        p = self.clf.predict_proba(np.array([_feat(self.state, home_id, away_id)]))[0]
        return float(p[0]), float(p[1]), float(p[2])

    def elo(self, team_id):
        return round(self.state.elo[team_id]) if self.state else None
