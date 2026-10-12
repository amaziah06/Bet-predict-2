"""Statistical core: time-decayed Poisson regression (attack, defence, home advantage,
team-specific home boost) + Dixon-Coles low-score correction, then every betting
market is derived from ONE scoreline probability matrix so markets stay consistent."""
from datetime import date
import math
import numpy as np
from scipy import sparse
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MAXG = 11  # scoreline grid 0..10


class StrengthModel:
    def __init__(self, half_life_days=240, alpha=0.004, home_scale=0.45):
        self.half_life = half_life_days
        self.alpha = alpha
        self.home_scale = home_scale  # <1 => team-specific home boost is shrunk harder
        self.rho = -0.05

    # ---- feature construction -------------------------------------------------
    def _row(self, scorer, conceder, scorer_is_home):
        n = self.n
        cols, vals = [self.idx[scorer]], [1.0]                       # attack
        cols.append(n + self.idx[conceder]); vals.append(1.0)         # defence
        if scorer_is_home:
            cols.append(2 * n); vals.append(1.0)                      # league-wide home advantage
            cols.append(2 * n + 1 + self.idx[scorer]); vals.append(self.home_scale)  # this team's extra home boost
        return cols, vals

    def _matrix(self, rows):
        indptr, indices, data = [0], [], []
        for cols, vals in rows:
            indices += cols; data += vals; indptr.append(len(indices))
        return sparse.csr_matrix((data, indices, indptr), shape=(len(rows), 3 * self.n + 1))

    def fit(self, matches, as_of: date):
        teams = sorted({m["home_id"] for m in matches} | {m["away_id"] for m in matches})
        self.idx = {t: i for i, t in enumerate(teams)}
        self.n = len(teams)
        rows, y, w = [], [], []
        for m in matches:
            age = (as_of - date.fromisoformat(m["date"])).days
            if age < 0:
                continue
            wt = 0.5 ** (age / self.half_life)
            rows.append(self._row(m["home_id"], m["away_id"], True)); y.append(m["hg"]); w.append(wt)
            rows.append(self._row(m["away_id"], m["home_id"], False)); y.append(m["ag"]); w.append(wt)
        X = self._matrix(rows)
        self.glm = PoissonRegressor(alpha=self.alpha, max_iter=500, fit_intercept=True)
        self.glm.fit(X, np.array(y), sample_weight=np.array(w))
        self._fit_rho(matches, as_of)
        return self

    def known(self, team_id):
        return team_id in self.idx

    def lambdas(self, home_id, away_id):
        """Expected goals (home, away). Unknown teams fall back to league-average strength."""
        def one(a, b, is_home):
            if a not in self.idx or b not in self.idx:
                tmp_a = a if a in self.idx else None
                # average-team fallback: use intercept (+ home adv)
                base = self.glm.intercept_ + (self.glm.coef_[2 * self.n] if is_home else 0)
                att = self.glm.coef_[self.idx[a]] if a in self.idx else 0
                dfn = self.glm.coef_[self.n + self.idx[b]] if b in self.idx else 0
                return float(np.exp(base + att + dfn))
            X = self._matrix([self._row(a, b, is_home)])
            return float(self.glm.predict(X)[0])
        return one(home_id, away_id, True), one(away_id, home_id, False)

    # ---- Dixon-Coles rho ----------------------------------------------------------
    def _fit_rho(self, matches, as_of):
        low = [m for m in matches if m["hg"] <= 1 and m["ag"] <= 1 and date.fromisoformat(m["date"]) <= as_of]
        if len(low) < 50:
            return
        lam = [self.lambdas(m["home_id"], m["away_id"]) for m in matches[-1500:]]
        best, best_ll = self.rho, -1e18
        for rho in np.arange(-0.20, 0.051, 0.01):
            ll = 0.0
            for m, (lh, la) in zip(matches[-1500:], lam):
                if m["hg"] <= 1 and m["ag"] <= 1:
                    ll += math.log(max(_tau(m["hg"], m["ag"], lh, la, rho), 1e-9))
            if ll > best_ll:
                best, best_ll = rho, ll
        self.rho = float(best)


def _tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0


def score_matrix(lh, la, rho=-0.05):
    ph = poisson.pmf(np.arange(MAXG), lh)
    pa = poisson.pmf(np.arange(MAXG), la)
    P = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            P[i, j] *= max(_tau(i, j, lh, la, rho), 0)
    return P / P.sum()


def rescale_1x2(P, target):
    """Reweight the scoreline matrix so its 1X2 totals equal `target` (home, draw, away)."""
    i, j = np.indices(P.shape)
    out = P.copy()
    for mask, t in ((i > j, target[0]), (i == j, target[1]), (i < j, target[2])):
        s = P[mask].sum()
        out[mask] = P[mask] * (t / s if s > 0 else 0)
    return out / out.sum()


# ------------------------------- markets ------------------------------------------
def _diff_dist(lh, la, n=9):
    ph, pa = poisson.pmf(np.arange(n), lh), poisson.pmf(np.arange(n), la)
    d = {}
    for i in range(n):
        for j in range(n):
            d[i - j] = d.get(i - j, 0) + ph[i] * pa[j]
    return d


def _sgn(x):
    return 0 if x == 0 else (1 if x > 0 else 2)  # 0=draw,1=home,2=away


def markets(P, lh, la, first_half_share=0.44):
    i, j = np.indices(P.shape)
    tot = i + j
    M = {}
    p1, px, p2 = P[i > j].sum(), P[i == j].sum(), P[i < j].sum()
    M["1X2"] = {"Home": p1, "Draw": px, "Away": p2}
    M["Double chance"] = {"1X": p1 + px, "12": p1 + p2, "X2": px + p2}
    M["Draw no bet"] = {"Home": p1 / (p1 + p2), "Away": p2 / (p1 + p2)}

    ou = {}
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        o = P[tot > line].sum()
        ou[f"Over {line}"] = o; ou[f"Under {line}"] = 1 - o
    M["Total goals over/under"] = ou
    M["Exact total goals"] = {str(k): P[tot == k].sum() for k in range(0, 6)} | {"6+": P[tot >= 6].sum()}
    M["Odd/Even total"] = {"Odd": P[tot % 2 == 1].sum(), "Even": P[tot % 2 == 0].sum()}

    btts = P[(i > 0) & (j > 0)].sum()
    M["Both teams to score"] = {"Yes": btts, "No": 1 - btts}
    M["BTTS & Over 2.5"] = {"Yes & Over": P[(i > 0) & (j > 0) & (tot > 2.5)].sum(),
                            "Yes & Under": P[(i > 0) & (j > 0) & (tot < 2.5)].sum(),
                            "No & Over": P[((i == 0) | (j == 0)) & (tot > 2.5)].sum(),
                            "No & Under": P[((i == 0) | (j == 0)) & (tot < 2.5)].sum()}
    M["Win to nil"] = {"Home": P[(i > 0) & (j == 0)].sum(), "Away": P[(j > 0) & (i == 0)].sum()}
    M["Clean sheet"] = {"Home": P[:, 0].sum(), "Away": P[0, :].sum()}
    M["Team to score"] = {"Home scores": P[1:, :].sum(), "Away scores": P[:, 1:].sum()}
    M["Team total goals"] = {f"Home Over {l}": P[i > l].sum() for l in (0.5, 1.5, 2.5)} | \
                            {f"Away Over {l}": P[j > l].sum() for l in (0.5, 1.5, 2.5)}
    M["Home/Away to score 2+"] = {"Home 2+": P[i >= 2].sum(), "Away 2+": P[j >= 2].sum()}

    # Winning margin
    mg = {}
    for k in (1, 2, 3):
        mg[f"Home by {k}" if k < 3 else "Home by 3+"] = P[(i - j == k)].sum() if k < 3 else P[(i - j >= 3)].sum()
        mg[f"Away by {k}" if k < 3 else "Away by 3+"] = P[(j - i == k)].sum() if k < 3 else P[(j - i >= 3)].sum()
    mg["Draw"] = px
    M["Winning margin"] = mg

    # Asian handicap on the home side (win / push / lose), halves and wholes
    ah = {}
    for h in (-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5):
        d = i - j + h
        win, push = P[d > 0].sum(), P[d == 0].sum()
        label = f"Home {h:+g}"
        ah[label] = {"win": win, "push": push, "lose": 1 - win - push}
    M["Asian handicap (home line)"] = ah

    # Correct score (top 8)
    flat = sorted(((P[a, b], f"{a}-{b}") for a in range(7) for b in range(7)), reverse=True)[:8]
    M["Correct score (top 8)"] = {s: p for p, s in flat}

    # Halves (independent Poisson halves, standard approximation)
    lh1, la1 = lh * first_half_share, la * first_half_share
    lh2, la2 = lh * (1 - first_half_share), la * (1 - first_half_share)
    d1, d2 = _diff_dist(lh1, la1), _diff_dist(lh2, la2)
    ht = {0: 0.0, 1: 0.0, 2: 0.0}
    htft = {}
    for a, pa_ in d1.items():
        ht[_sgn(a)] += pa_
        for b, pb in d2.items():
            k = (_sgn(a), _sgn(a + b))
            htft[k] = htft.get(k, 0) + pa_ * pb
    nm = {0: "Draw", 1: "Home", 2: "Away"}
    s = sum(ht.values())
    M["Half-time result"] = {nm[k]: v / s for k, v in ht.items()}
    st = sum(htft.values())
    M["Half-time / Full-time"] = {f"{nm[a]}/{nm[b]}": v / st for (a, b), v in sorted(htft.items(), key=lambda kv: -kv[1])}
    p0 = math.exp(-(lh1 + la1)); p01 = p0 * (1 + (lh1 + la1))
    M["1st half goals"] = {"Over 0.5": 1 - p0, "Under 0.5": p0, "Over 1.5": 1 - p01, "Under 1.5": p01}
    return M


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    return o
