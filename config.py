import os
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CACHE_DIR = DATA_DIR / "cache"
PRED_DIR = DATA_DIR / "predictions"
for d in (DATA_DIR, CACHE_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Get a key at https://dashboard.api-football.com (free plan: 100 requests/day)
API_KEY = os.getenv("API_FOOTBALL_KEY", "")
FD_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN", "")   # free token from https://www.football-data.org/client/register
# Provider: football-data.org if its token is set, else API-Football if its key is set, else demo
PROVIDER = "footballdata" if FD_TOKEN else ("apifootball" if API_KEY else "demo")
DEMO = os.getenv("DEMO", "").lower() in ("1", "true", "yes") or PROVIDER == "demo"

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


# ---- accounts / time zone -------------------------------------------------------
# Persistent database for accounts (free Postgres, e.g. neon.tech). Without it accounts are kept in a
# local SQLite file, which a free host wipes on every restart.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
SESSION_DAYS = 30

# "Today" is decided in this time zone (not the server's UTC), so the day flips at your midnight.
APP_TZ = os.getenv("APP_TZ", "Africa/Lagos")


def tzinfo():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(APP_TZ)
    except Exception:
        print(f"[warn] time zone {APP_TZ!r} unavailable, using UTC")
        return timezone.utc


def today_local():
    return datetime.now(tzinfo()).date()


# competitions with league tables / top scorers (football-data.org codes)
TABLE_CODES = {"PL": 39, "PD": 140, "SA": 135, "PPL": 94, "CL": 2}
