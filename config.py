import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CACHE_DIR = DATA_DIR / "cache"
PRED_DIR = DATA_DIR / "predictions"
for d in (DATA_DIR, CACHE_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Get a key at https://dashboard.api-football.com (free plan: 100 requests/day)
API_KEY = os.getenv("API_FOOTBALL_KEY", "")
DEMO = os.getenv("DEMO", "").lower() in ("1", "true", "yes") or not API_KEY

# API-Football league ids
LEAGUES = {
    39: {"name": "Premier League", "country": "England", "cup": False},
    140: {"name": "La Liga", "country": "Spain", "cup": False},
    135: {"name": "Serie A", "country": "Italy", "cup": False},
    94: {"name": "Primeira Liga", "country": "Portugal", "cup": False},
    203: {"name": "Süper Lig", "country": "Turkey", "cup": False},
    2: {"name": "Champions League", "country": "Europe", "cup": True},
    3: {"name": "Europa League", "country": "Europe", "cup": True},
    848: {"name": "Conference League", "country": "Europe", "cup": True},
}

# A season is named after the year it starts (2026 = 2026/27)
def current_season(today) -> int:
    return today.year if today.month >= 7 else today.year - 1

HISTORY_SEASONS_BACK = 1      # also load previous season for more data
HALF_LIFE_DAYS = 240          # time-decay of old matches in the strength model
FORM_MATCHES = 8
H2H_MATCHES = 10
ML_WEIGHT = 0.20              # statistics dominate; ML only nudges 1X2
REFRESH_HOUR_LOCAL = 6        # daily rebuild hour (server local time)
