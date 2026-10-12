---
title: BetPredict
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# BetPredict

Daily match-outcome predictions for the Premier League, La Liga, Serie A, Primeira Liga,
Süper Lig, Champions League, Europa League and Conference League.

## Run

```bash
pip install -r requirements.txt
DEMO=1 uvicorn app.main:app --reload          # simulated data, no key needed
# real data (free key from https://dashboard.api-football.com, 100 requests/day):
API_FOOTBALL_KEY=your_key uvicorn app.main:app
```
Open http://localhost:8000. Predictions rebuild automatically every day at 06:00 server time.

## How it predicts
1. **Strength model**: time-decayed Poisson regression (attack, defence, league home advantage,
   per-team home boost) on two seasons of results, plus a Dixon-Coles low-score correction.
2. **Context adjustments** (small, capped): last-8 form vs expectation, head-to-head, injuries
   weighted by each missing player's share of team goals + assists.
3. **ML nudge**: gradient-boosted 1X2 classifier (Elo, form, home/away form), 20% weight.
4. **One scoreline matrix** feeds every market: 1X2, double chance, DNB, over/under, BTTS,
   correct score, Asian handicap, team totals, clean sheet, win to nil, winning margin,
   half-time, HT/FT, odd/even, exact goals.
5. **Value bets**: model probability vs bookmaker odds (when the API returns odds).

Not covered: corners, cards, player props (need separate data and models).

## Check it before trusting it
`python -m scripts.backtest 8` runs a walk-forward test (log-loss, Brier, accuracy vs a naive
baseline). Run it on real data first; the model only has an edge if it beats the baseline,
and beating bookmaker closing odds is much harder than that.

## Notes
- Europa/Conference and Süper Lig coverage needs API-Football; football-data.org's free tier lacks them.
- Odds are 1X2, over/under 1.5-3.5 and BTTS only; other markets show probabilities only.
- Probabilities are estimates, not guarantees. 18+, bet responsibly.
