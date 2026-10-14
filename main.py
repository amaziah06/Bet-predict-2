import asyncio
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, db, service, store
from .config import (API_KEY, BASE_DIR, DEMO, FD_TOKEN, LEAGUES, PROVIDER, REFRESH_HOUR_LOCAL, SESSION_DAYS,
                     APP_TZ, TABLE_CODES, current_season, today_local)

COOKIE = "bp_session"
STATIC = BASE_DIR / "app" / "static"
_lock = asyncio.Lock()


# ----------------------------------- data building -------------------------------------
async def _build(day: date):
    async with _lock:  # one build at a time (it is CPU/IO heavy)
        try:
            return await asyncio.to_thread(service.build, day)
        except RuntimeError as e:
            raise HTTPException(503, str(e))


async def _patch(cached):
    async with _lock:
        try:
            return await asyncio.to_thread(service.refresh_statuses, cached)
        except Exception as e:
            print("[warn] status refresh failed:", e)
            return cached


async def _get_day(d: date):
    """The day's predictions. Today's data is kept fresh: rebuilt if old, statuses/scores refreshed often."""
    ds = d.isoformat()
    cached = service.get_cached(ds)
    if cached is not None and d == today_local():
        if service.age_seconds(cached.get("generated_at")) > 10 * 3600:
            cached = None
        elif not cached["matches"] and service.age_seconds(cached.get("generated_at")) > 1800:
            cached = None
        elif cached["matches"] and service.age_seconds(cached.get("status_updated_at")) > 600:
            cached = await _patch(cached)
    if cached is None:
        cached = await _build(d)
    return cached


_settle_lock = asyncio.Lock()
_last_settle = 0.0


async def _settle_if_due(min_gap=300):
    """Fetch final scores for finished games (accuracy log + slips). At most once per `min_gap` seconds."""
    global _last_settle
    if time.time() - _last_settle < min_gap:
        return
    _last_settle = time.time()
    async with _settle_lock:
        try:
            await asyncio.to_thread(service.settle_all)
        except Exception as e:
            print("[warn] settling failed:", e)


async def _daily_loop():
    while True:
        today = today_local()
        for d in (today, today + timedelta(days=1)):
            cached = service.get_cached(d.isoformat())
            if cached is None or (d == today and service.age_seconds(cached.get("generated_at")) > 10 * 3600):
                try:
                    await _build(d)
                except Exception as e:
                    print("[daily] build failed:", e)
        await _settle_if_due(0)
        now = datetime.now()
        nxt = now.replace(hour=REFRESH_HOUR_LOCAL, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        await asyncio.sleep(max(60, (nxt - now).total_seconds()))


@asynccontextmanager
async def lifespan(app):
    try:
        db.init()
        auth.purge_expired()
    except Exception as e:
        print("[warn] database setup failed:", e)
    task = asyncio.create_task(_daily_loop())
    yield
    task.cancel()


app = FastAPI(title="BetPredict", lifespan=lifespan)


@app.middleware("http")
async def no_store(request: Request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ------------------------------------ throttling ---------------------------------------
_hits: dict = {}


def _recent(key, window):
    now = time.time()
    lst = [t for t in _hits.get(key, []) if now - t < window]
    _hits[key] = lst
    if len(_hits) > 5000:
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > 3600]:
            _hits.pop(k, None)
    return lst


def _blocked(key, limit, window):
    return len(_recent(key, window)) >= limit


def _add_hit(key):
    _hits.setdefault(key, []).append(time.time())


def _client_ip(request: Request):
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


# --------------------------------------- auth ------------------------------------------
class Credentials(BaseModel):
    email: str
    password: str


def _set_cookie(request: Request, response: Response, token: str):
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    response.set_cookie(COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax",
                        secure=(proto == "https"), path="/")


def _user_or_none(request: Request):
    try:
        return auth.user_for_token(request.cookies.get(COOKIE))
    except Exception as e:
        print("[warn] session lookup failed:", e)
        return None


def current_user(request: Request):
    user = _user_or_none(request)
    if not user:
        raise HTTPException(401, "Please log in.")
    return user


DB_DOWN = "The account database is not reachable right now. Try again in a minute."


@app.post("/api/signup")
def signup(body: Credentials, request: Request, response: Response):
    ip = _client_ip(request)
    if _blocked(f"signup:{ip}", 10, 3600):
        raise HTTPException(429, "Too many sign-ups from this network. Try again later.")
    _add_hit(f"signup:{ip}")
    email = auth.normalize_email(body.email)
    if not auth.valid_email(email):
        raise HTTPException(400, "Enter a valid email address.")
    if not 8 <= len(body.password) <= 128:
        raise HTTPException(400, "Choose a password of 8 to 128 characters.")
    try:
        uid = auth.create_user(email, body.password)
        token = auth.new_session(uid)
    except auth.EmailExists:
        raise HTTPException(409, "An account with this email already exists. Log in instead.")
    except Exception as e:
        print("[error] signup:", e)
        raise HTTPException(503, DB_DOWN)
    _set_cookie(request, response, token)
    return {"email": email}


@app.post("/api/login")
def login(body: Credentials, request: Request, response: Response):
    email = auth.normalize_email(body.email)
    key = f"login:{_client_ip(request)}:{email}"
    if _blocked(key, 8, 600):
        raise HTTPException(429, "Too many failed attempts. Wait 10 minutes and try again.")
    try:
        user = auth.authenticate(email, body.password)
        if user:
            token = auth.new_session(user["id"])
    except Exception as e:
        print("[error] login:", e)
        raise HTTPException(503, DB_DOWN)
    if not user:
        _add_hit(key)
        raise HTTPException(401, "Wrong email or password.")
    _set_cookie(request, response, token)
    return {"email": user["email"]}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    try:
        auth.delete_session(request.cookies.get(COOKIE))
    except Exception as e:
        print("[warn] logout:", e)
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(current_user)):
    try:
        open_slip = store.slip_count(user["id"])
    except Exception:
        open_slip = 0
    return {"email": user["email"], "today": today_local().isoformat(), "slip_open": open_slip}


@app.get("/api/status")
def status():
    return {"demo_mode": DEMO, "storage_warning": db.WARNING, "leagues": {k: v["name"] for k, v in LEAGUES.items()}}


# ------------------------------------- predictions -------------------------------------
def _parse_day(day):
    today = today_local()
    if not day:
        return today
    try:
        d = date.fromisoformat(day)
    except ValueError:
        raise HTTPException(400, "day must be YYYY-MM-DD")
    if not today - timedelta(days=1) <= d <= today + timedelta(days=3):
        raise HTTPException(400, "Only yesterday, today and the next 3 days are available.")
    return d


@app.get("/api/predictions")
async def predictions(day: str | None = None, user=Depends(current_user)):
    return await _get_day(_parse_day(day))


@app.get("/api/surest")
async def surest(day: str | None = None, user=Depends(current_user)):
    data = await _get_day(_parse_day(day))
    picks = service.pick_surest(data["matches"], 2)
    try:  # show the picks that were saved before kick-off, so this page matches the Accuracy page
        logged = await asyncio.to_thread(store.logged_surest, data["date"])
        by_id = {x["fixture_id"]: x for x in data["matches"]}
        saved = [service.pick_from_logged(by_id[l["fixture_id"]], l) for l in logged if l["fixture_id"] in by_id]
        if saved:
            picks = saved
    except Exception as e:
        print("[warn] saved picks unavailable:", e)
    return {"date": data["date"], "demo": data.get("demo"), "notice": data.get("notice"),
            "n_matches": len(data["matches"]), "picks": picks}


@app.post("/api/refresh")
async def refresh(day: str | None = None, user=Depends(current_user)):
    d = _parse_day(day)
    cached = service.get_cached(d.isoformat())
    if cached and service.age_seconds(cached.get("generated_at")) < 900:  # protects the free API quota
        return await _get_day(d)
    return await _build(d)


# --------------------------------------- bet slip --------------------------------------
class SlipAdd(BaseModel):
    fixture_id: int
    date: str
    market: str
    key: str


@app.post("/api/slip")
async def slip_add(body: SlipAdd, user=Depends(current_user)):
    data = await _get_day(_parse_day(body.date))
    match = next((x for x in data["matches"] if x["fixture_id"] == body.fixture_id), None)
    if not match:
        raise HTTPException(404, "Game not found.")
    if match.get("status") != "upcoming" or store._past(match["kickoff"], 0):
        raise HTTPException(400, "This game has already started.")
    opt = next((o for o in service.slip_options(match) if o["market"] == body.market and o["key"] == body.key), None)
    if not opt:
        raise HTTPException(400, "Unknown bet.")
    try:
        res = await asyncio.to_thread(store.slip_add, user["id"], {**match, "date": data["date"]}, opt)
        n = await asyncio.to_thread(store.slip_count, user["id"])
    except Exception as e:
        print("[error] slip add:", e)
        raise HTTPException(503, DB_DOWN)
    if res == "duplicate":
        raise HTTPException(409, "That bet is already on your slip.")
    if res == "full":
        raise HTTPException(400, f"Your slip is full ({store.SLIP_LIMIT} open bets). Remove one first.")
    return {"ok": True, "open": n, "label": opt["label"]}


@app.get("/api/slip")
async def slip_get(user=Depends(current_user)):
    await _settle_if_due(120)
    try:
        items = await asyncio.to_thread(store.slip_list, user["id"])
    except Exception as e:
        print("[error] slip list:", e)
        raise HTTPException(503, DB_DOWN)
    pending = [i for i in items if i["status"] == "pending"]
    prod = 1.0
    for i in pending:
        prod *= i["probability"] or 0
    return {"items": items, "open": len(pending),
            "combined_probability": round(prod, 4) if pending else None,
            "fair_odds": round(1 / prod, 2) if pending and prod > 0 else None,
            "won": sum(1 for i in items if i["status"] == "won"),
            "lost": sum(1 for i in items if i["status"] == "lost")}


@app.delete("/api/slip/{item_id}")
async def slip_delete(item_id: int, user=Depends(current_user)):
    ok = await asyncio.to_thread(store.slip_delete, user["id"], item_id)
    if not ok:
        raise HTTPException(404, "Not found.")
    return {"ok": True}


@app.post("/api/slip/clear")
async def slip_clear(user=Depends(current_user)):
    return {"removed": await asyncio.to_thread(store.slip_clear_settled, user["id"])}


# ------------------------------------- accuracy / tables -------------------------------
@app.get("/api/accuracy")
async def accuracy(user=Depends(current_user)):
    asyncio.create_task(_settle_if_due(300))  # refresh results in the background; shown on the next visit
    try:
        return await asyncio.to_thread(store.accuracy)
    except Exception as e:
        print("[error] accuracy:", e)
        raise HTTPException(503, DB_DOWN)


@app.get("/api/standings")
async def standings(comp: str = "PL", user=Depends(current_user)):
    lid = TABLE_CODES.get(comp.upper())
    if not lid:
        raise HTTPException(400, "Unknown competition.")
    try:
        data = await asyncio.to_thread(service.table_for, lid)
    except Exception as e:
        print("[warn] standings:", e)
        raise HTTPException(503, "Could not load the table right now. Try again in a minute.")
    return {"comp": comp.upper(), "name": LEAGUES[lid]["name"], **data,
            "comps": [{"code": c, "name": LEAGUES[l]["name"]} for c, l in TABLE_CODES.items()]}


_diag = {"t": 0.0, "v": None}


@app.get("/api/diagnose")
def diagnose():
    """Setup check: which provider and database are active (secrets are never shown)."""
    if _diag["v"] and time.time() - _diag["t"] < 30:
        return _diag["v"]
    import requests
    info = {"provider": PROVIDER, "demo_mode": DEMO, "football_data_token_set": bool(FD_TOKEN),
            "football_data_token_length": len(FD_TOKEN), "api_football_key_set": bool(API_KEY),
            "time_zone": APP_TZ, "server_today": today_local().isoformat(),
            "database": db.kind(), "storage_warning": db.WARNING, "tests": []}
    try:
        with db.connect() as c:
            db.run(c, "SELECT 1")
        info["database_ok"] = True
    except Exception as e:
        info["database_ok"] = False
        info["database_error"] = str(e)[:150]
    if FD_TOKEN:
        season = current_season(today_local())
        for code, sea in (("PL", season), ("PL", season - 1), ("CL", season)):
            try:
                r = requests.get(f"https://api.football-data.org/v4/competitions/{code}/matches",
                                 params={"season": sea, "status": "FINISHED"},
                                 headers={"X-Auth-Token": FD_TOKEN}, timeout=25)
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                info["tests"].append({"competition": code, "season": sea, "http_status": r.status_code,
                                      "matches_returned": len(body.get("matches", [])),
                                      "message": (body.get("message") or "")[:200]})
            except Exception as e:
                info["tests"].append({"competition": code, "season": sea, "error": str(e)[:200]})
    _diag.update(t=time.time(), v=info)
    return info


# ----------------------------------------- pages ---------------------------------------
def _page(request: Request, name: str, need_login: bool):
    logged = _user_or_none(request) is not None
    if need_login and not logged:
        return RedirectResponse("/login", status_code=302)
    if not need_login and logged:
        return RedirectResponse("/games", status_code=302)
    return FileResponse(STATIC / name)


@app.get("/")
def index(request: Request):
    return RedirectResponse("/games" if _user_or_none(request) else "/login", status_code=302)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/login")
def login_page(request: Request):
    return _page(request, "login.html", need_login=False)


@app.get("/games")
def games_page(request: Request):
    return _page(request, "games.html", need_login=True)


@app.get("/surest")
def surest_page(request: Request):
    return _page(request, "surest.html", need_login=True)


@app.get("/tables")
def tables_page(request: Request):
    return _page(request, "tables.html", need_login=True)


@app.get("/slip")
def slip_page(request: Request):
    return _page(request, "slip.html", need_login=True)


@app.get("/accuracy")
def accuracy_page(request: Request):
    return _page(request, "accuracy.html", need_login=True)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
