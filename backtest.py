"""Walk-forward backtest: for each test week, fit only on matches before it, predict the
week, and score log-loss / Brier / accuracy for stats-only, ML-only and the blend,
against a naive baseline. Run:  python -m scripts.backtest [weeks]"""
import sys
from datetime import date, timedelta
import numpy as np

from app import service
from app.config import ML_WEIGHT
from app.model import StrengthModel, score_matrix, rescale_1x2
from app.ml import MLModel


def outcome(m):
    return 0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2


def main(weeks=8, step_days=7):
    hist = service.load_history(date.today() + timedelta(days=1))
    if not hist:
        print("no history loaded"); return
    today = date.fromisoformat(max(m["date"] for m in hist)) + timedelta(days=1)  # anchor to newest data
    print(f"{len(hist)} matches loaded, last match {today - timedelta(days=1)}")
    rows = {"stats": [], "ml": [], "blend": [], "base": []}
    ys = []
    freq = np.bincount([outcome(m) for m in hist], minlength=3) / max(len(hist), 1)
    for w in range(weeks, 0, -1):
        start = today - timedelta(days=step_days * w)
        end = start + timedelta(days=step_days)
        train = [m for m in hist if date.fromisoformat(m["date"]) < start]
        test = [m for m in hist if start <= date.fromisoformat(m["date"]) < end]
        if len(train) < 500 or not test:
            continue
        sm = StrengthModel().fit(train, start)
        ml = MLModel().fit(train)
        for m in test:
            if not (sm.known(m["home_id"]) and sm.known(m["away_id"])):
                continue
            lh, la = sm.lambdas(m["home_id"], m["away_id"])
            P = score_matrix(lh, la, sm.rho)
            i, j = np.indices(P.shape)
            s = np.array([P[i > j].sum(), P[i == j].sum(), P[i < j].sum()])
            mp = ml.predict(m["home_id"], m["away_id"])
            mlv = np.array(mp) if mp else s
            rows["stats"].append(s); rows["ml"].append(mlv)
            rows["blend"].append((1 - ML_WEIGHT) * s + ML_WEIGHT * mlv); rows["base"].append(freq)
            ys.append(outcome(m))
    y = np.array(ys)
    if not len(y):
        print("not enough data"); return
    print(f"{len(y)} test matches")
    print(f"{'model':8s} {'logloss':>8s} {'brier':>8s} {'acc':>7s}")
    for k, v in rows.items():
        p = np.clip(np.array(v), 1e-6, 1)
        ll = -np.mean(np.log(p[np.arange(len(y)), y]))
        oh = np.eye(3)[y]
        br = np.mean(np.sum((p - oh) ** 2, axis=1))
        acc = np.mean(p.argmax(axis=1) == y)
        print(f"{k:8s} {ll:8.4f} {br:8.4f} {acc:7.1%}")
    print("Lower log-loss/Brier is better. If 'stats' does not beat 'base', the model has no signal.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
