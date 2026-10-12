import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import service
from .config import BASE_DIR, DEMO, LEAGUES, REFRESH_HOUR_LOCAL

_lock = asyncio.Lock()


async def _build(day: date):
    async with _lock:  # one build at a time (it is CPU/IO heavy)
        return await asyncio.to_thread(service.build, day)


async def _daily_loop():
    while True:
        today = date.today()
        for d in (today, today + timedelta(days=1)):
            if service.get_cached(d.isoformat()) is None:
                try:
                    await _build(d)
                except Exception as e:
                    print("[daily] build failed:", e)
        now = datetime.now()
        nxt = now.replace(hour=REFRESH_HOUR_LOCAL, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        await asyncio.sleep(max(60, (nxt - now).total_seconds()))
        try:
            await _build(date.today())  # force a fresh rebuild each morning
        except Exception as e:
            print("[daily] refresh failed:", e)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(_daily_loop())
    yield
    task.cancel()


app = FastAPI(title="BetPredict", lifespan=lifespan)


@app.get("/api/status")
def status():
    return {"demo_mode": DEMO, "leagues": {k: v["name"] for k, v in LEAGUES.items()}}


@app.get("/api/predictions")
async def predictions(day: str | None = None):
    ds = day or date.today().isoformat()
    try:
        d = date.fromisoformat(ds)
    except ValueError:
        raise HTTPException(400, "day must be YYYY-MM-DD")
    cached = service.get_cached(ds)
    if cached is None:
        cached = await _build(d)
    return cached


@app.post("/api/refresh")
async def refresh(day: str | None = None):
    d = date.fromisoformat(day) if day else date.today()
    return await _build(d)


app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "app" / "static" / "index.html")
