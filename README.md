# BetPredict

Daily football predictions with accounts (email + password).

## Settings (environment variables)
- `FOOTBALL_DATA_TOKEN` - free token from football-data.org
- `DATABASE_URL` - free Postgres connection string (e.g. from neon.tech) so accounts survive restarts
- `APP_TZ` - optional, defaults to `Africa/Lagos`; the day changes at midnight in this time zone

## Run locally
```
pip install -r requirements.txt
DEMO=1 uvicorn app.main:app
```

## Pages
- `/login` - log in / sign up
- `/games` - every game of the day with all markets
- `/surest` - the 2 surest games of the day, one best bet each
- `/tables` - league tables and top scorers
- `/slip` - your saved bets (tracker only), settled automatically
- `/accuracy` - permanent record of pre-kick-off predictions vs real results

Predictions are model estimates, not guarantees.
