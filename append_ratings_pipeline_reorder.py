from __future__ import annotations

# Aurora Arcade — Ratings pipeline (PATCHED, rate-limit–aware)
# - Prepass: use PEGI column if present; pre-1994 rule for most pre-ESRB era platforms (except SEGA VRC + 3DO)
# - Cache-first: resume from SQLite cache
# - API order (least → most rate-limited): IGDB (prefetch all boards) → RAWG (ESRB only) → Wikidata (SPARQL) → GiantBomb (last)
# - For IGDB/RAWG/Wikidata we fetch once per row (covers all boards we can) to minimize tokens
# - GiantBomb is only used if others failed
# - Threaded, checkpointing, and periodic summaries

from typing import Any, Dict, List, Optional, Set, Tuple

from dataclasses import dataclass

import argparse, time, json, re, sqlite3, logging, datetime, random, unicodedata, threading, itertools
from collections import defaultdict, deque
from urllib.parse import urlparse

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ratings")

# ===== Shared helpers =====
class APICallCounter:
    """Thread-safe counter for tracking outbound API calls."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def incr(self, name: str) -> None:
        if not name:
            return
        with self._lock:
            self._counts[name] += 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)


class ProgressReporter:
    """Emit richer progress updates with ETA and API call counts."""

    def __init__(
        self,
        total: int,
        row_interval: int = 0,
        time_interval: float = 45.0,
        lock: Optional[threading.Lock] = None,
    ) -> None:
        self._total = max(1, int(total))
        self._row_interval = max(0, int(row_interval))
        self._time_interval = max(5.0, float(time_interval))
        self._lock = lock
        self._start = time.time()
        self._last_emit = 0.0

    def maybe_emit(
        self,
        attempted: int,
        successes: int,
        source_counts: Dict[str, int],
        api_counts: Dict[str, int],
    ) -> None:
        now = time.time()
        if attempted <= 0:
            return
        if self._row_interval and (attempted % self._row_interval == 0):
            should_emit = True
        else:
            should_emit = (now - self._last_emit) >= self._time_interval
        if not should_emit:
            return
        self._last_emit = now
        elapsed = max(1e-6, now - self._start)
        rate = attempted / elapsed
        remain = max(0, self._total - attempted)
        eta_minutes = (remain / rate) / 60.0 if rate > 0 else None
        coverage = (successes / attempted) * 100.0 if attempted else 0.0
        api_bits = ", ".join(f"{k}={v}" for k, v in sorted(api_counts.items())) if api_counts else "none"
        src_bits = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())) if source_counts else "none"
        msg = (
            "Progress: %d/%d rows (%.1f%%) | coverage %.1f%% (%d with ≥1 rating) | rate %.2f rows/s"
            % (attempted, self._total, (attempted / self._total) * 100.0, coverage, successes, rate)
        )
        if eta_minutes is not None:
            msg += f" | ETA ≈ {eta_minutes:.1f} min"
        msg += f" | Sources {src_bits} | API calls {api_bits}"
        if self._lock:
            with self._lock:
                log.info(msg)
        else:
            log.info(msg)

# ===== Pretty printing =====
def _pretty_table(headers: List[str], rows: List[List[Any]]) -> str:
    cols = len(headers)
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i in range(cols):
            if i < len(r):
                widths[i] = max(widths[i], len(str(r[i])))
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = f"+{sep}+"
    def _row(cells):
        return "|" + "|".join(" " + str(cells[i])[:widths[i]].ljust(widths[i]) + " " for i in range(cols)) + "|"
    out = [sep, _row(headers), sep]
    for r in rows: out.append(_row(r))
    out.append(sep)
    return "\n".join(out)

def _print_section(title: str, verbose: bool, lock: Optional[threading.Lock]=None):
    if not verbose: return
    if lock:
        with lock:
            print("\n" + "=" * 80, flush=True); print(title, flush=True); print("=" * 80, flush=True)
    else:
        print("\n" + "=" * 80, flush=True); print(title, flush=True); print("=" * 80, flush=True)

def _print_kv(kv: Dict[str, Any], verbose: bool, title: Optional[str]=None, lock: Optional[threading.Lock]=None):
    if not verbose: return
    if lock:
        with lock:
            if title: print(title, flush=True)
            if not kv: print("(none)", flush=True); return
            headers = ["Field","Value"]; rows = [[k,v] for k,v in kv.items()]
            print(_pretty_table(headers, rows), flush=True)
    else:
        if title: print(title, flush=True)
        if not kv: print("(none)", flush=True); return
        headers = ["Field","Value"]; rows = [[k,v] for k,v in kv.items()]
        print(_pretty_table(headers, rows), flush=True)

def _print_rows(headers: List[str], rows: List[List[Any]], verbose: bool, title: Optional[str]=None, lock: Optional[threading.Lock]=None):
    if not verbose: return
    if lock:
        with lock:
            if title: print(title, flush=True)
            if not rows: print("(none)", flush=True); return
            print(_pretty_table(headers, rows), flush=True)
    else:
        if title: print(title, flush=True)
        if not rows: print("(none)", flush=True); return
        print(_pretty_table(headers, rows), flush=True)

# ===== Minimal progress line printer =====
def _progress_line(platform: str, title: str, api: str, status: str, verbose: bool=False, lock: Optional[threading.Lock]=None):
    if not verbose: return
    line = f"[{platform}] | {title} | {api} | {status}"
    if lock:
        with lock:
            print(line, flush=True)
    else:
        print(line, flush=True)

# ===== Fuzzy matching =====
try:
    from rapidfuzz import fuzz
except Exception:
    from difflib import SequenceMatcher
    class _F:
        @staticmethod
        def token_sort_ratio(a,b):
            def _n(x): return ' '.join(sorted(x.split()))
            sm = SequenceMatcher(None, _n(a), _n(b)); return int(round(sm.ratio()*100))
        @staticmethod
        def partial_ratio(a,b):
            sm = SequenceMatcher(None, a, b); return int(round(sm.ratio()*100))
    fuzz = _F()

_ROMAN = {'ix':9,'viii':8,'vii':7,'vi':6,'iv':4,'iii':3,'ii':2,'i':1}
_ROMAN_RE = re.compile(r'\b(' + '|'.join(sorted(_ROMAN, key=len, reverse=True)) + r')\b')

@lru_cache(maxsize=8192)
def _norm_cached(text: str) -> str:
    s = text.lower()
    s = unicodedata.normalize('NFKD', s)
    s = re.sub(r"[®™©]", "", s)
    s = re.sub(r"[^a-z0-9\s\-:]", " ", s)
    s = re.sub(r"\b(the|a|an|and|of|for|edition|remastered|definitive|hd|goty|collection|complete|ultimate)\b", " ", s)
    s = _ROMAN_RE.sub(lambda m: str(_ROMAN.get(m.group(1), m.group(1))), s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm(text: str) -> str:
    return _norm_cached("" if text is None else str(text))

@lru_cache(maxsize=8192)
def _token_tuple(text: str) -> Tuple[str, ...]:
    return tuple(t for t in _norm_cached(text).split() if t)

def token_set(text: str) -> Set[str]:
    return set(_token_tuple("" if text is None else str(text)))

# ===== Platform helpers =====
PLATFORM_ALIASES = {
    "gc":"Nintendo GameCube","gen":"Sega Genesis","md":"Sega Mega Drive","nes":"Nintendo Entertainment System",
    "snes":"Super Nintendo Entertainment System","ng":"Neo Geo","ngcd":"Neo Geo CD","ps":"PlayStation","ps1":"PlayStation",
    "ps2":"PlayStation 2","ps3":"PlayStation 3","ps4":"PlayStation 4","ps5":"PlayStation 5","psp":"PlayStation Portable",
    "vita":"PlayStation Vita","xbox":"Xbox","x360":"Xbox 360","xb1":"Xbox One","xsx":"Xbox Series X|S","switch":"Nintendo Switch",
    "3do":"3DO Interactive Multiplayer","pc":"PC (Microsoft Windows)","2600":"Atari 2600",
}

DIGITAL_ALIASES = {
    "Xbox 360": ["Xbox Live Arcade"],
    "Xbox One": ["Xbox Live Arcade", "Xbox Store"],
    "Xbox Series X|S": ["Xbox Store"],
    "PlayStation 3": ["PlayStation Network (PS3)", "PlayStation Network"],
    "PlayStation 4": ["PlayStation Network (PS4)", "PlayStation Network"],
    "PlayStation 5": ["PlayStation Network (PS5)", "PlayStation Network"],
    "PlayStation Portable": ["PlayStation Network (PSP)", "PlayStation Network"],
    "PlayStation Vita": ["PlayStation Network (Vita)", "PlayStation Network"],
    "Wii": ["WiiWare"],
    "Wii U": ["Nintendo eShop (Wii U)", "Nintendo eShop"],
    "Nintendo Switch": ["Nintendo eShop"],
    "Nintendo 3DS": ["Nintendo eShop (3DS)", "Nintendo eShop"],
}

def _canonical_platform_name(name: Optional[str]) -> Optional[str]:
    if not name: return name
    key = str(name).strip().lower()
    return PLATFORM_ALIASES.get(key, name)

def platform_name_matches(platform_names: List[str], target_platform: Optional[str]) -> bool:
    if not target_platform: return True
    tset = token_set(str(target_platform))
    for nm in platform_names or []:
        if len(tset & token_set(nm)) >= max(1, min(len(tset), len(token_set(nm))) // 2):
            return True
    return False

# Special cases for pre-1994 exemption logic

def _platform_is_sega_vrc(plat: str) -> bool:
    names = [
        "Master System","Sega Master System",
        "Genesis","Mega Drive","Sega Genesis",
        "Game Gear","Sega Game Gear",
        "Sega CD","Mega-CD","Mega CD",
        "32X","Sega 32X"
    ]
    try:
        if platform_name_matches(names, plat): return True
    except Exception:
        pass
    p = (plat or "").strip().lower()
    aliases = {"gen","md","megadrive","gg","scd","megacd","32x","mastersystem","sms","sg1000"}
    return p in aliases

def _platform_is_3do(plat: str) -> bool:
    try:
        return platform_name_matches(["3DO"], plat)
    except Exception:
        return (str(plat or "").strip().lower() == "3do")

# ===== Cache (SQLite) =====
db_lock = threading.Lock()

def get_db(path="ratings_cache.sqlite"):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)""")
    conn.commit()
    return conn

def cache_get(conn, key):
    with db_lock:
        c = conn.cursor(); c.execute("SELECT v FROM kv WHERE k=?", (key,))
        row = c.fetchone()
    return None if row is None else row[0]

def cache_put(conn, key, value):
    with db_lock:
        c = conn.cursor(); c.execute("INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)", (key, value))
        conn.commit()

# Final-result cache helpers (per title/platform/year)

def _final_key(title: str, platform: str, year: Optional[int]) -> str:
    return f"final::{norm(title)}::{norm(platform)}::{year or ''}"

def final_cache_get(conn, title: str, platform: str, year: Optional[int]) -> Optional[Dict[str, Any]]:
    key = _final_key(title, platform, year)
    raw = cache_get(conn, key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

def final_cache_put(conn, title: str, platform: str, year: Optional[int], result: Dict[str, Any]) -> None:
    key = _final_key(title, platform, year)
    try:
        cache_put(conn, key, json.dumps(result))
    except Exception:
        pass

# Convenience: mark negative result in a separate table (optional noop here)

def final_cache_get_negative(conn, title, platform, year):
    try:
        with db_lock:
            cur = conn.cursor()
            year_val: Optional[int]
            try:
                year_val = int(year) if year not in (None, "") else None
            except Exception:
                year_val = None
            if year_val is None:
                cur.execute(
                    "SELECT payload FROM ratings_cache WHERE title=? AND platform=? AND year IS NULL ORDER BY ts DESC LIMIT 1",
                    (str(title), str(platform)),
                )
            else:
                cur.execute(
                    "SELECT payload FROM ratings_cache WHERE title=? AND platform=? AND year=? ORDER BY ts DESC LIMIT 1",
                    (str(title), str(platform), year_val),
                )
            row = cur.fetchone()
    except sqlite3.OperationalError:
        return None
    except Exception:
        return None
    if not row or not row[0]:
        return None
    try:
        payload = json.loads(row[0]) if isinstance(row[0], str) else None
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return None
    if payload.get("__no_rating"):
        return payload
    return None

def final_cache_put_negative(conn, title, platform, year):
    try:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS ratings_cache (title TEXT, platform TEXT, year INT, payload TEXT, ts INT)'
        )
        conn.execute(
            'INSERT OR REPLACE INTO ratings_cache (title, platform, year, payload, ts) VALUES (?,?,?,?,strftime("%s","now"))',
            (str(title), str(platform), int(year) if (year not in (None, "")) else None, json.dumps({"__checked": True, "__no_rating": True}))
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

# ===== Thread-safe rate limiting =====
class RPSLimiter:
    def __init__(self, rps: float):
        self.interval = 1.0 / max(1e-6, rps)
        self.next_ok = 0.0
        self.lock = threading.Lock()
    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_ok:
                time.sleep(self.next_ok - now)
                now = time.time()
            # introduce tiny jitter
            self.next_ok = max(now, self.next_ok) + self.interval * (0.9 + 0.2*random.random())

class HourlyQuotaLimiter:
    def __init__(self, max_calls: int, window_seconds: int = 3600):
        self.max_calls = max_calls
        self.window = window_seconds
        self.events = deque()
        self.lock = threading.Lock()
    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                while self.events and now - self.events[0] > self.window:
                    self.events.popleft()
                if len(self.events) < self.max_calls:
                    self.events.append(now)
                    return
                wait = self.window - (now - self.events[0])
            if wait > 0:
                time.sleep(min(wait, 60))

class MinSpacingLimiter:
    def __init__(self, min_seconds: float):
        self.min = float(min_seconds)
        self.last = 0.0
        self.lock = threading.Lock()
    def wait(self):
        with self.lock:
            now = time.time()
            delta = now - self.last
            if delta < self.min:
                time.sleep(self.min - delta)
            self.last = time.time()

# ===== In-flight deduplication =====
class InFlightDeduper:
    """Coordinate workers so each (title, platform, year) triggers APIs once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}

    def acquire(self, key: str) -> Tuple[bool, threading.Event]:
        """Return (is_owner, event). Non-owners should wait on the event."""

        with self._lock:
            evt = self._inflight.get(key)
            if evt is not None:
                return False, evt
            evt = threading.Event()
            self._inflight[key] = evt
            return True, evt

    def release(self, key: str) -> None:
        with self._lock:
            evt = self._inflight.pop(key, None)
        if evt is not None:
            evt.set()


@dataclass
class IGDBPrefetchResult:
    """Result returned by the IGDB batch prefetch queue."""

    fulfilled: bool
    via_batch: bool
    game: Optional[Dict[str, Any]] = None


# ===== IGDB client =====
class IGDBClient:
    def __init__(self, client_id: str, client_secret: str, cache_conn,
                 rl_global: RPSLimiter, rl_api: RPSLimiter,
                 print_lock: Optional[threading.Lock]=None, verbose: bool=False,
                 api_counter: Optional[APICallCounter]=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.cache = cache_conn
        self.verbose = verbose
        self.print_lock = print_lock
        self.rl_global = rl_global
        self.rl_api = rl_api
        self.api_counter = api_counter
        self._session = requests.Session()
        self._session_lock = threading.Lock()

        self._tok_lock = threading.Lock()
        self._token = None
        self._token_expiry = 0.0

        self._search_cache: Dict[str, List[int]] = {}
        self._search_cache_lock = threading.Lock()
        self._game_cache: Dict[int, Dict[str, Any]] = {}
        self._game_cache_lock = threading.Lock()

        try:
            payload = cache_get(self.cache, "igdb_token_payload")
            if payload:
                data = json.loads(payload)
                self._token = data.get("access_token")
                self._token_expiry = float(data.get("expiry", 0.0))
        except Exception:
            pass

    def _log(self, msg):
        if self.verbose:
            if self.print_lock:
                with self.print_lock: print(msg, flush=True)
            else:
                print(msg, flush=True)

    def _count(self) -> None:
        if self.api_counter:
            self.api_counter.incr("IGDB")

    def _save_token(self, access_token: str, expires_in: int):
        expiry = time.time() + max(0, int(expires_in) - 60)
        self._token = access_token
        self._token_expiry = expiry
        cache_put(self.cache, "igdb_token_payload",
                  json.dumps({"access_token": access_token, "expiry": expiry}))

    def _fetch_token(self):
        self._log("[IGDB] Requesting OAuth token…")
        self.rl_global.wait(); self.rl_api.wait()
        self._count()
        with self._session_lock:
            r = self._session.post(
                "https://id.twitch.tv/oauth2/token",
                params={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                },
                timeout=20,
            )
        r.raise_for_status()
        data = r.json()
        self._save_token(data["access_token"], data.get("expires_in", 3600))
        self._log("[IGDB] Token acquired.")

    def _ensure_token(self, force=False):
        with self._tok_lock:
            if force or (not self._token) or (time.time() >= self._token_expiry):
                self._fetch_token()

    def _post_once(self, endpoint: str, body: str):
        headers = {"Client-ID": self.client_id, "Authorization": f"Bearer {self._token}"}
        url = f"https://api.igdb.com/v4/{endpoint}"
        self._log(f"[IGDB] POST /{endpoint} body={body[:120]}…")
        self.rl_global.wait(); self.rl_api.wait()
        self._count()
        with self._session_lock:
            r = self._session.post(url, data=body, headers=headers, timeout=30)
        return r

    def _post(self, endpoint: str, body: str):
        self._ensure_token(force=False)
        r = self._post_once(endpoint, body)
        if r.status_code == 401:
            self._log("[IGDB] 401 received; refreshing token and retrying once…")
            with self._tok_lock:
                self._token = None
                self._token_expiry = 0.0
                cache_put(self.cache, "igdb_token_payload", json.dumps({"access_token": "", "expiry": 0.0}))
            self._ensure_token(force=True)
            r = self._post_once(endpoint, body)
        if r.status_code >= 400:
            log.warning("IGDB %s error (%d): %s | body=%s", endpoint, r.status_code, r.text[:300], body)
        r.raise_for_status()
        return r.json()

    def _platform_id_for_name(self, plat_name: str) -> Optional[int]:
        plat_name = _canonical_platform_name(plat_name)
        if not plat_name: return None
        key = f"igdb_plat_id::{plat_name}"
        cached = cache_get(self.cache, key)
        if cached:
            try:
                self._log(f"[IGDB] Cached platform id '{plat_name}' = {cached}")
                return int(cached)
            except Exception: pass
        q = str(plat_name).replace('"','')
        res = self._post("platforms", f'fields id,name,abbreviation; search "{q}"; limit 15;')
        best = None; best_s = -1
        for r in res:
            cand = r.get("name") or ""
            ts = fuzz.token_sort_ratio(norm(plat_name), norm(cand))
            pr = fuzz.partial_ratio(norm(plat_name), norm(cand))
            score = min(ts, pr)
            if score > best_s: best, best_s = r, score
        if best and best_s >= 70:
            pid = int(best["id"]); cache_put(self.cache, key, str(pid))
            self._log(f"[IGDB] Platform '{plat_name}' → id {pid}")
            return pid
        cache_put(self.cache, key, "")
        self._log(f"[IGDB] Could not resolve platform '{plat_name}'")
        return None

    def _platform_ids_for_base_with_aliases(self, base_platform_name: Optional[str]) -> Set[int]:
        ids: Set[int] = set()
        if not base_platform_name: return ids
        base = _canonical_platform_name(base_platform_name)
        if base:
            pid = self._platform_id_for_name(base)
            if pid: ids.add(pid)
        for alias in DIGITAL_ALIASES.get(base, []):
            pid = self._platform_id_for_name(alias)
            if pid: ids.add(pid)
        return ids

    def _score_game(self, target_title: str, g: Dict[str, Any], year: Optional[int], threshold: int) -> int:
        s = 0; name_g = g.get("name","")
        if same_title(target_title, name_g, threshold): s += 5
        else:
            a = norm(target_title); b = norm(name_g)
            s += min(fuzz.token_sort_ratio(a,b), fuzz.partial_ratio(a,b))//40
        if year:
            y = None
            if g.get('release_dates'):
                yrs = [d.get('y') for d in g['release_dates'] if d.get('y')]
                if yrs: y = min(yrs)
            if not y and g.get('first_release_date'):
                try: y = pd.to_datetime(g['first_release_date'], unit='s', utc=True).year
                except Exception: y = None
            if y and abs(int(y) - int(year)) <= 1: s += 1
        return s

    def _fetch_games_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        unique_ids = list(dict.fromkeys(int(i) for i in ids if i))
        results: Dict[int, Dict[str, Any]] = {}
        missing: List[int] = []

        for gid in unique_ids:
            cached_game = None
            with self._game_cache_lock:
                cached_game = self._game_cache.get(gid)
            if cached_game is None:
                raw = cache_get(self.cache, f"igdb_game::{gid}")
                if raw:
                    try:
                        cached_game = json.loads(raw)
                    except Exception:
                        cached_game = None
                    else:
                        with self._game_cache_lock:
                            self._game_cache[gid] = cached_game
            if cached_game is not None:
                results[gid] = cached_game
            else:
                missing.append(gid)

        if missing:
            fields = "id,name,first_release_date,release_dates.y,platforms,age_ratings"
            where = "id = ({})".format(",".join(map(str, missing)))
            fetched = self._post("games", f"fields {fields}; where {where}; limit 50;")
            for g in fetched or []:
                try:
                    gid = int(g.get("id"))
                except Exception:
                    continue
                if gid:
                    results[gid] = g
                    cache_put(self.cache, f"igdb_game::{gid}", json.dumps(g))
                    with self._game_cache_lock:
                        self._game_cache[gid] = g

        ordered: List[Dict[str, Any]] = []
        for gid in unique_ids:
            if gid in results:
                ordered.append(results[gid])
        return ordered

    def batch_search_games(self, search_payloads: List[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Resolve multiple title searches via a single /multiquery call."""

        out: Dict[str, Optional[Dict[str, Any]]] = {}
        if not search_payloads:
            return out

        entries: List[Dict[str, Any]] = []
        need_query: List[Dict[str, Any]] = []

        for payload in search_payloads:
            key = payload.get("key")
            name_raw = (payload.get("name") or "").strip()
            if key is None:
                continue
            if not name_raw:
                out[str(key)] = None
                continue

            platform_name = payload.get("platform_name")
            year = payload.get("year")
            threshold = int(payload.get("threshold", 78))

            norm_name = norm(name_raw)
            platform_norm = norm(platform_name or "")
            cache_key = f"igdb_search::{norm_name}::{platform_norm}::{year or ''}::{threshold}"

            cached = cache_get(self.cache, cache_key)
            if cached is not None:
                try:
                    parsed = json.loads(cached)
                except Exception:
                    parsed = None
                if parsed:
                    out[str(key)] = parsed
                else:
                    out[str(key)] = None
                continue

            entry = {
                "key": str(key),
                "name": name_raw,
                "platform_name": platform_name,
                "year": year,
                "threshold": threshold,
                "norm_name": norm_name,
                "cache_key": cache_key,
                "plat_ids": self._platform_ids_for_base_with_aliases(platform_name) if platform_name else set(),
                "search_ids": None,
            }

            search_ids: Optional[List[int]] = None
            with self._search_cache_lock:
                if norm_name in self._search_cache:
                    search_ids = list(self._search_cache[norm_name])
            if search_ids is None:
                cached_ids = cache_get(self.cache, f"igdb_search_ids::{norm_name}")
                if cached_ids:
                    try:
                        parsed_ids = json.loads(cached_ids)
                        if isinstance(parsed_ids, list):
                            search_ids = [int(x) for x in parsed_ids if x]
                    except Exception:
                        search_ids = None

            if search_ids:
                entry["search_ids"] = list(dict.fromkeys(int(i) for i in search_ids if i))
            else:
                need_query.append(entry)

            entries.append(entry)

        if need_query:
            body_bits: List[str] = []
            for idx, entry in enumerate(need_query):
                q = entry["name"].replace('"', "").replace("\n", " ").strip()
                alias = f"q{idx}"
                body_bits.append(
                    f'query {alias} "search" {{ fields name,game; search "{q}"; limit 25; }}'
                )
            if body_bits:
                try:
                    self._log(f"[IGDB] Batch searching {len(body_bits)} titles via multiquery…")
                    response = self._post("multiquery", "\n".join(body_bits))
                except Exception as e:
                    raise
                else:
                    for idx, entry in enumerate(need_query):
                        rows = []
                        if isinstance(response, list) and idx < len(response):
                            rows = response[idx].get("result") or response[idx].get("results") or []
                        ids = [int(r.get("game")) for r in rows if r.get("game")]
                        entry["search_ids"] = list(dict.fromkeys(ids))
                        cache_put(self.cache, f"igdb_search_ids::{entry['norm_name']}", json.dumps(entry["search_ids"]))
                        with self._search_cache_lock:
                            self._search_cache[entry["norm_name"]] = list(entry["search_ids"])

        all_ids: List[int] = []
        for entry in entries:
            ids = entry.get("search_ids") or []
            entry["search_ids"] = list(dict.fromkeys(int(i) for i in ids if i))
            all_ids.extend(entry["search_ids"])

        games_by_id: Dict[int, Dict[str, Any]] = {}
        if all_ids:
            fetched_games = self._fetch_games_by_ids(all_ids)
            for g in fetched_games:
                try:
                    games_by_id[int(g.get("id"))] = g
                except Exception:
                    continue

        for entry in entries:
            key = entry["key"]
            cache_key = entry["cache_key"]
            ids = entry.get("search_ids") or []
            if not ids:
                cache_put(self.cache, cache_key, json.dumps({}))
                out[key] = None
                continue

            games = [games_by_id.get(i) for i in ids if games_by_id.get(i)]
            if not games:
                cache_put(self.cache, cache_key, json.dumps({}))
                out[key] = None
                continue

            plat_ids: Set[int] = entry.get("plat_ids") or set()
            best = None
            best_s = -999
            for g in games:
                plats = set(int(p) for p in (g.get("platforms") or []))
                if plat_ids and not (plats & plat_ids):
                    continue
                sc = self._score_game(entry["name"], g, entry["year"], entry["threshold"])
                if sc > best_s:
                    best, best_s = g, sc
            if best and same_title(entry["name"], best.get("name", ""), entry["threshold"]):
                cache_put(self.cache, cache_key, json.dumps(best))
                out[key] = best
                continue

            best2 = None
            best2_s = -999
            for g in games:
                sc = self._score_game(entry["name"], g, entry["year"], entry["threshold"])
                if sc > best2_s:
                    best2, best2_s = g, sc
            if best2 and same_title(entry["name"], best2.get("name", ""), entry["threshold"]):
                cache_put(self.cache, cache_key, json.dumps(best2))
                out[key] = best2
            else:
                cache_put(self.cache, cache_key, json.dumps({}))
                out[key] = None

        return out

    def search_game_tiered(self, name: str, platform_name: Optional[str], year: Optional[int], threshold: int) -> Optional[Dict[str, Any]]:
        if not name: return None
        q = name.strip()
        plat_ids = self._platform_ids_for_base_with_aliases(platform_name) if platform_name else set()

        norm_name = norm(name)
        cache_key = f"igdb_search::{norm_name}::{norm(platform_name or '')}::{year or ''}::{threshold}"
        cached = cache_get(self.cache, cache_key)
        if cached is not None:
            try:
                data = json.loads(cached)
                if data:
                    return data
                return None
            except Exception:
                pass

        # Search by title with cached search ID reuse
        search_ids: Optional[List[int]] = None
        with self._search_cache_lock:
            if norm_name in self._search_cache:
                search_ids = list(self._search_cache[norm_name])
        if search_ids is None:
            cached_ids = cache_get(self.cache, f"igdb_search_ids::{norm_name}")
            if cached_ids:
                try:
                    parsed = json.loads(cached_ids)
                    if isinstance(parsed, list):
                        search_ids = [int(x) for x in parsed if x]
                except Exception:
                    search_ids = None
        if search_ids is None:
            self._log(f"[IGDB] Searching for '{name}' (plat={platform_name or '—'}, year={year or '—'})")
            res = self._post("search", f'fields name,game; search "{q}"; limit 25;')
            search_ids = [int(r.get("game")) for r in res if r.get("game")]
            cache_put(self.cache, f"igdb_search_ids::{norm_name}", json.dumps(search_ids))
            with self._search_cache_lock:
                self._search_cache[norm_name] = list(search_ids)
        else:
            self._log(f"[IGDB] Using cached search results for '{name}'")

        game_ids = list(dict.fromkeys(search_ids or []))
        if not game_ids:
            cache_put(self.cache, cache_key, json.dumps({}))
            return None
        games = self._fetch_games_by_ids(game_ids)

        # Pass A: platform-scoped (base + digital aliases)
        best = None; best_s = -999
        for g in games:
            plats = set(int(p) for p in (g.get("platforms") or []))
            if plat_ids and not (plats & plat_ids):
                continue
            sc = self._score_game(name, g, year, threshold)
            if sc > best_s: best, best_s = g, sc
        if best and same_title(name, best.get("name",""), threshold):
            cache_put(self.cache, cache_key, json.dumps(best))
            return best

        # Pass B: unscoped (fallback)
        self._log("[IGDB] No solid platform-scoped match; retrying without platform constraint…")
        best2 = None; best2_s = -999
        for g in games:
            sc = self._score_game(name, g, year, threshold)
            if sc > best2_s: best2, best2_s = g, sc
        result = None
        if best2 and same_title(name, best2.get("name",""), threshold):
            result = best2
        cache_put(self.cache, cache_key, json.dumps(result or {}))
        return result

    def age_ratings_for(self, game_id: int) -> List[Dict[str, Any]]:
        if not game_id: return []
        cache_key = f"igdb_age::{int(game_id)}"
        cached = cache_get(self.cache, cache_key)
        if cached is not None:
            try:
                data = json.loads(cached)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        g = self._post("games", f"fields age_ratings; where id = {game_id}; limit 1;")
        if not g or not g[0].get("age_ratings"):
            cache_put(self.cache, cache_key, json.dumps([]))
            return []
        rating_ids = g[0]["age_ratings"]
        rs = self._post(
            "age_ratings",
            f"fields id,category,rating,content_descriptions,rating_cover_url; where id = ({','.join(map(str, rating_ids))}); limit 50;"
        )
        all_desc_ids = []
        for r in rs: all_desc_ids.extend(r.get("content_descriptions") or [])
        desc_map = {}
        if all_desc_ids:
            uniq = list(dict.fromkeys(all_desc_ids))
            ds = self._post(
                "age_rating_content_descriptions",
                f"fields id,description; where id = ({','.join(map(str, uniq))}); limit 200;"
            )
            for d in ds: desc_map[d["id"]] = d.get("description","")
        for r in rs:
            r["content_description_texts"] = [desc_map.get(i, "") for i in (r.get("content_descriptions") or []) if desc_map.get(i)]
        cache_put(self.cache, cache_key, json.dumps(rs))
        return rs

@dataclass
class _IGDBBatchRequest:
    key: str
    title: str
    platform: Optional[str]
    year: Optional[int]
    threshold: int
    event: threading.Event
    result: Optional[Dict[str, Any]] = None
    fulfilled: bool = False
    via_batch: bool = False
    error: Optional[Exception] = None


class IGDBBatcher:
    """Coordinate batched IGDB searches across worker threads."""

    def __init__(self, client: IGDBClient, batch_size: int, wait_seconds: float = 0.05) -> None:
        self.client = client
        self.batch_size = max(1, int(batch_size))
        self.wait_seconds = max(0.0, float(wait_seconds))
        self._lock = threading.Lock()
        self._queue: deque[_IGDBBatchRequest] = deque()
        self._counter = itertools.count(1)

    def fetch(self, title: str, platform: Optional[str], year: Optional[int], threshold: int) -> IGDBPrefetchResult:
        if not self.client or self.batch_size <= 1:
            return IGDBPrefetchResult(fulfilled=False, via_batch=False, game=None)

        req = _IGDBBatchRequest(
            key=f"req{next(self._counter)}",
            title=title,
            platform=platform,
            year=year,
            threshold=threshold,
            event=threading.Event(),
        )

        batch: Optional[List[_IGDBBatchRequest]] = None
        with self._lock:
            self._queue.append(req)
            if len(self._queue) >= self.batch_size:
                take = min(self.batch_size, len(self._queue))
                batch = [self._queue.popleft() for _ in range(take)]

        if batch:
            self._execute_batch(batch)

        if not req.event.wait(self.wait_seconds):
            with self._lock:
                if req in self._queue:
                    self._queue.remove(req)
                    batch = [req]
                else:
                    batch = None
            if batch:
                self._execute_batch(batch)
            if not req.event.is_set():
                req.event.wait()

        if not req.fulfilled or req.error is not None:
            return IGDBPrefetchResult(fulfilled=False, via_batch=False, game=None)

        game_payload = req.result if isinstance(req.result, dict) and req.result else None
        return IGDBPrefetchResult(fulfilled=True, via_batch=req.via_batch, game=game_payload)

    def _execute_batch(self, batch: List[_IGDBBatchRequest]) -> None:
        payloads = [
            {
                "key": req.key,
                "name": req.title,
                "platform_name": req.platform,
                "year": req.year,
                "threshold": req.threshold,
            }
            for req in batch
        ]
        try:
            mapping = self.client.batch_search_games(payloads)
        except Exception as exc:
            log.warning("IGDB batch search failed (%d rows): %s", len(batch), exc)
            for req in batch:
                req.error = exc
                req.fulfilled = False
                req.via_batch = False
                req.event.set()
            return

        for req in batch:
            req.result = mapping.get(req.key)
            req.fulfilled = True
            req.via_batch = True
            req.event.set()


IGDB_ORG = {1:"ESRB",2:"PEGI",3:"CERO",4:"USK",5:"GRAC",6:"CLASS_IND",7:"ACB"}
IGDB_RATING_NAME = {
    1:"PEGI 3",2:"PEGI 7",3:"PEGI 12",4:"PEGI 16",5:"PEGI 18",6:"RP",
    7:"EC",8:"E",9:"E10+",10:"T",11:"M",12:"AO",
    13:"CERO A",14:"CERO B",15:"CERO C",16:"CERO D",17:"CERO Z",
    18:"USK 0",19:"USK 6",20:"USK 12",21:"USK 16",22:"USK 18",
    23:"GRAC ALL",24:"GRAC 12",25:"GRAC 15",26:"GRAC 18",27:"GRAC TESTING",
    28:"CLASS_IND L",29:"CLASS_IND 10",30:"CLASS_IND 12",31:"CLASS_IND 14",32:"CLASS_IND 16",33:"CLASS_IND 18",
    34:"ACB G",35:"ACB PG",36:"ACB M",37:"ACB MA15+",38:"ACB R18",39:"ACB RC"
}

def ratings_from_igdb_full(game: Dict[str, Any], igdb: IGDBClient) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not game or not game.get("id"): return out
    for r in igdb.age_ratings_for(game["id"]):
        org_name = IGDB_ORG.get(r.get("category"))
        label = IGDB_RATING_NAME.get(r.get("rating"))
        if org_name and label:
            col = f"rating_{org_name.lower()}"
            if not out.get(col):
                out[col] = label
                descs = r.get("content_description_texts") or []
                if descs: out[f"{col}_descriptors"] = "; ".join(sorted(set(d for d in descs if d)))
                out[f"{col}_source"] = "IGDB"
    return out

# ===== Wikidata =====
def wikidata_ratings(title: str, platform_hint: Optional[str], year: Optional[int],
                     user_agent: str, cache_conn, rl_global: RPSLimiter, rl_api: RPSLimiter,
                     verbose: bool=False, api_counter: Optional[APICallCounter]=None) -> Dict[str,str]:
    key = f"wd::{title}::{platform_hint}::{year}"
    cached = cache_get(cache_conn, key)
    if cached is not None:
        try: return json.loads(cached)
        except Exception: pass

    headers = {"Accept":"application/sparql-results+json"}
    if user_agent: headers["User-Agent"] = user_agent
    title_escaped = title.replace('"','\\"')
    plat_filter = f'?item wdt:P400 ?platform . ?platform rdfs:label ?plat_label . FILTER(CONTAINS(LCASE(STR(?plat_label)), LCASE("{platform_hint}"))) .' if platform_hint else ""
    year_filter = f"FILTER (year(?date) >= {int(year)-1} && year(?date) <= {int(year)+1}) ." if year else ""
    query = f"""
    SELECT ?item ?itemLabel ?esrb ?esrbLabel ?pegi ?pegiLabel ?cero ?ceroLabel ?usk ?uskLabel WHERE {{
      ?item rdfs:label "{title_escaped}"@en .
      OPTIONAL {{ ?item wdt:P852 ?esrb . }}
      OPTIONAL {{ ?item wdt:P908 ?pegi . }}
      OPTIONAL {{ ?item wdt:P853 ?cero . }}
      OPTIONAL {{ ?item wdt:P914 ?usk . }}
      OPTIONAL {{ ?item wdt:P577 ?date . }}
      {year_filter}
      {plat_filter}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT 5
    """
    rl_global.wait(); rl_api.wait()
    if api_counter:
        api_counter.incr("Wikidata")
    r = requests.get("https://query.wikidata.org/sparql", params={"query": query, "format":"json"}, headers=headers, timeout=30)
    out: Dict[str,str] = {}
    if r.status_code == 200:
        for b in r.json().get("results",{}).get("bindings",[]):
            if "pegiLabel" in b and not out.get("PEGI"): out["PEGI"] = b["pegiLabel"]["value"]
            if "esrbLabel" in b and not out.get("ESRB"): out["ESRB"] = b["esrbLabel"]["value"]
            if "ceroLabel" in b and not out.get("CERO"): out["CERO"] = b["ceroLabel"]["value"]
            if "uskLabel"  in b and not out.get("USK"):  out["USK"]  = b["uskLabel"]["value"]
    else:
        log.warning("Wikidata HTTP %d: %s", r.status_code, r.text[:200])
    cache_put(cache_conn, key, json.dumps(out))
    return out

# ===== GiantBomb =====
class GiantBombClient:
    def __init__(self, api_key: str, user_agent: Optional[str], cache_conn,
                 rl_global: RPSLimiter, rps_api: RPSLimiter,
                 hourly_search: HourlyQuotaLimiter, hourly_details: HourlyQuotaLimiter,
                 spacing: MinSpacingLimiter, print_lock: Optional[threading.Lock]=None, verbose: bool=False,
                 api_counter: Optional[APICallCounter]=None):
        self.api_key = api_key
        self.user_agent = (user_agent or "RatingsPipeline/1.0")
        self.cache = cache_conn
        self.base = "https://www.giantbomb.com/api"
        self.verbose = verbose
        self.print_lock = print_lock
        self.rl_global = rl_global
        self.rps_api = rps_api
        self.hourly_search = hourly_search
        self.hourly_details = hourly_details
        self.spacing = spacing
        self.api_counter = api_counter
        self._session = requests.Session()
        self._session_lock = threading.Lock()

    def _request(self, path: str, params: Dict[str, Any], which: str, max_retries: int = 6) -> Dict[str, Any]:
        headers = {"User-Agent": self.user_agent, "Accept":"application/json"}
        q = dict(params or {}); q["api_key"] = self.api_key; q["format"] = "json"
        url = f"{self.base}/{path.lstrip('/')}"
        backoff = 15.0
        for attempt in range(1, max_retries+1):
            self.rl_global.wait()
            self.rps_api.wait()
            self.spacing.wait()
            (self.hourly_search if which == "search" else self.hourly_details).wait()
            if self.api_counter:
                self.api_counter.incr("GiantBomb")

            with self._session_lock:
                r = self._session.get(url, headers=headers, params=q, timeout=30, allow_redirects=True)
            txt = r.text[:200].replace("\n"," ")

            if r.status_code in (420, 429, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                wait = float(ra) if (ra and ra.isdigit()) else backoff
                log.warning("GiantBomb %s throttle (%d). Sleeping %.1fs. Attempt %d/%d", path, r.status_code, wait, attempt, max_retries)
                time.sleep(wait); backoff = min(backoff*1.7, 180.0); continue

            if r.status_code >= 400:
                log.warning("GiantBomb %s error (%d): %s | url=%s", path, r.status_code, txt, r.url)
                return {}

            try:
                data = r.json()
            except Exception:
                log.warning("GiantBomb %s JSON parse failed; first 200 chars: %s", path, txt)
                return {}

            if isinstance(data, dict) and data.get("status_code") == 107:
                wait = backoff
                log.warning("GiantBomb %s status_code 107 (slow down). Sleeping %.1fs. Attempt %d/%d", path, wait, attempt, max_retries)
                time.sleep(wait); backoff = min(backoff*1.7, 180.0); continue

            return data
        return {}

    @staticmethod
    def _strip_punct(s: str) -> str:
        return re.sub(r"[^\w\s]", " ", s).strip()

    def search_game(self, name: str, year: Optional[int]) -> List[Dict[str, Any]]:
        def _do(qname: str, cache_key: str):
            cached = cache_get(self.cache, cache_key)
            if cached:
                try:
                    data = json.loads(cached)
                    return data.get("results") or []
                except Exception:
                    pass
            data = self._request("search/", {
                "query": qname, "resources":"game", "limit": 8,
                "field_list":"name,api_detail_url,original_release_date,guid"
            }, which="search")
            cache_put(self.cache, cache_key, json.dumps(data))
            return data.get("results") or []

        key1 = f"gb_search::{norm(name)}::{year or ''}"
        results = _do(name, key1)
        if results:
            return results
        cleaned = self._strip_punct(name)
        if cleaned and cleaned != name:
            key2 = f"gb_search_np::{norm(cleaned)}::{year or ''}"
            results2 = _do(cleaned, key2)
            return results2
        return results

    def _path_from_api_detail_url(self, api_detail_url: str) -> Optional[str]:
        if not api_detail_url: return None
        p = urlparse(api_detail_url); path = p.path or ""
        if path.startswith("/api/"): path = path[len("/api/"):]
        return path.lstrip("/") if path else None

    def game_details(self, api_detail_url: str) -> Dict[str, Any]:
        sub = self._path_from_api_detail_url(api_detail_url)
        if not sub: return {}
        data = self._request(sub, {"field_list":"name,platforms,original_game_rating,guid"}, which="details")
        return data.get("results",{}) or {}

def parse_giantbomb_ratings(ratings_list: Optional[List[Dict[str, Any]]]) -> Dict[str,str]:
    out: Dict[str,str] = {}
    if not ratings_list: return out
    for r in ratings_list:
        name = (r.get("name") or "").strip(); lower = name.lower()
        if "pegi" in lower:
            m = re.search(r'pegi[:\s]*([0-9]{1,2})', lower)
            out.setdefault("rating_pegi", f"PEGI {m.group(1)}" if m else "PEGI"); out.setdefault("rating_pegi_source","GiantBomb")
        elif "esrb" in lower:
            m = re.search(r'esrb[:\s]*([a-z0-9\+\-]+)', lower); label = (m.group(1).upper() if m else None)
            table = {"EC":"EC","E":"E","E10+":"E10+","T":"T","M":"M","AO":"AO","RP":"RP"}
            out.setdefault("rating_esrb", table.get(label, label or "ESRB")); out.setdefault("rating_esrb_source","GiantBomb")
        elif "cero" in lower:
            m = re.search(r'cero[:\s]*([abcdz])', lower)
            out.setdefault("rating_cero", f"CERO {m.group(1).upper()}" if m else "CERO"); out.setdefault("rating_cero_source","GiantBomb")
        elif "usk" in lower:
            m = re.search(r'usk[:\s]*([0-9]{1,2})', lower)
            out.setdefault("rating_usk", f"USK {m.group(1)}" if m else "USK"); out.setdefault("rating_usk_source","GiantBomb")
        elif "acb" in lower or "oflc" in lower or "au " in lower:
            out.setdefault("rating_acb","ACB"); out.setdefault("rating_acb_source","GiantBomb")
        elif "djctq" in lower or "classind" in lower or "class-ind" in lower:
            out.setdefault("rating_class_ind","CLASS_IND"); out.setdefault("rating_class_ind_source","GiantBomb")
    return out

# ===== RAWG (ESRB only) =====
class RAWGClient:
    def __init__(self, api_key: str, cache_conn, rl_global: RPSLimiter, rl_api: RPSLimiter,
                 api_counter: Optional[APICallCounter]=None):
        self.api_key = api_key; self.cache = cache_conn
        self.base = "https://api.rawg.io/api"
        self.rl_global = rl_global; self.rl_api = rl_api
        self.api_counter = api_counter
        self._session = requests.Session()
        self._session_lock = threading.Lock()
    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params); params["key"] = self.api_key
        self.rl_global.wait(); self.rl_api.wait()
        if self.api_counter:
            self.api_counter.incr("RAWG")
        with self._session_lock:
            r = self._session.get(f"{self.base}/{path.lstrip('/')}", params=params, timeout=30)
        if r.status_code >= 400:
            log.warning("RAWG %s error (%d): %s | url=%s", path, r.status_code, r.text[:200], r.url)
        r.raise_for_status()
        return r.json()
    def search(self, name: str, year: Optional[int]) -> List[Dict[str, Any]]:
        cache_key = f"rawg_search::{norm(name)}::{year or ''}"
        cached = cache_get(self.cache, cache_key)
        if cached is not None:
            try:
                data = json.loads(cached)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        params = {"search": name, "page_size": 5}
        if year: params["search_precise"] = True
        data = self._get("games", params)
        results = data.get("results") or []
        cache_put(self.cache, cache_key, json.dumps(results))
        return results
    def game_details(self, id_or_slug: Any) -> Dict[str, Any]:
        cache_key = f"rawg_game::{id_or_slug}"
        cached = cache_get(self.cache, cache_key)
        if cached is not None:
            try:
                data = json.loads(cached)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        try:
            data = self._get(f"games/{id_or_slug}", {})
            cache_put(self.cache, cache_key, json.dumps(data))
            return data
        except Exception:
            cache_put(self.cache, cache_key, json.dumps({}))
            return {}

def ratings_from_rawg(result: Dict[str, Any], plat_name: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not result: return out
    if plat_name:
        names = [p.get("platform",{}).get("name","") for p in (result.get("platforms") or [])]
        def _matches(names, target):
            tt = token_set(str(_canonical_platform_name(target)))
            for nm in names:
                if len(tt & token_set(nm)) >= max(1, min(len(tt), len(token_set(nm))) // 2):
                    return True
            return False
        if not _matches(names, _canonical_platform_name(plat_name)):
            return out
    esrb = (result.get("esrb_rating") or {}).get("name")
    if esrb:
        out["rating_esrb"] = esrb; out["rating_esrb_source"] = "RAWG"
    return out

# ===== Utilities =====

def _is_empty_or_nan(val) -> bool:
    try:
        if pd.isna(val):
            return True
    except Exception:
        pass
    s = str(val).strip().lower()
    return (s == "") or (s in {"nan","none","null","n/a","na"})

def _normalize_pegi_to_int(val) -> Optional[int]:
    if _is_empty_or_nan(val): return None
    m = re.search(r'(?:pegi\s*)?(\d{1,2})(?:\.0+)?', str(val).strip(), flags=re.I)
    if m:
        try: return int(m.group(1))
        except Exception: return None
    m = re.fullmatch(r'PEGI\s*(3|7|12|16|18)', str(val).strip(), flags=re.I)
    if m: return int(m.group(1))
    return None

def _board_to_numeric(board: str, value: str) -> Tuple[Optional[int], str]:
    b = (board or "").lower()
    v = (value or "").strip()
    if b == "pegi":
        n = _normalize_pegi_to_int(v); return (n, "PEGI")
    if b == "usk":
        m = re.search(r'(\d{1,2})', v); return ((int(m.group(1)) if m else None), "USK")
    if b == "esrb":
        table = {"EC":3,"E":3,"E10+":10,"T":13,"M":17,"AO":18}
        lab = v.upper().replace(" ", "")
        return (table.get(lab), "ESRB")
    if b == "cero":
        m = re.search(r'([ABCDZ])', v.upper()); mapc = {"A":3,"B":12,"C":15,"D":17,"Z":18}
        return ((mapc.get(m.group(1)) if m else None), "CERO")
    if b == "sega_vrc":
        lab = v.upper()
        if "MA-17" in lab or "MA17" in lab: return (17, "SEGA_VRC")
        if "MA-13" in lab or "MA13" in lab: return (13, "SEGA_VRC")
        if "GA" in lab or "G A" in lab: return (3, "SEGA_VRC")
        m = re.search(r'(\d{1,2})', lab); return ((int(m.group(1)) if m else None), "SEGA_VRC")
    if b == "3do":
        m = re.search(r'(\d{1,2})', v); return ((int(m.group(1)) if m else None), "3DO")
    m = re.search(r'(\d{1,2})', v); return ((int(m.group(1)) if m else None), (board.upper() if board else "UNKNOWN"))

BOARD_COL = {
    "pegi":"rating_pegi","esrb":"rating_esrb","cero":"rating_cero","usk":"rating_usk","3do":"rating_3do","sega_vrc":"rating_sega_vrc"
}

# ===== Title matching convenience =====

def same_title(a: str, b: str, threshold: int = 78) -> bool:
    a1, b1 = norm(a), norm(b)
    ts = fuzz.token_sort_ratio(a1, b1); pr = fuzz.partial_ratio(a1, b1)
    if min(ts, pr) < threshold: return False
    return token_set(a1) == token_set(b1)

# ===== Year helpers =====

def _extract_year_from_value_pre(v) -> Optional[int]:
    try:
        if v is None: return None
        s = str(v).strip()
        f = float(s); y = int(f)
        if 1900 <= y <= 2100: return y
    except Exception:
        pass
    try:
        y = pd.to_datetime(str(v), errors="coerce").year
        if pd.notna(y): return int(y)
    except Exception:
        pass
    return None

def _extract_year_from_row_any(df: pd.DataFrame, row: pd.Series, args) -> Optional[int]:
    cands: List[str] = []
    try:
        if getattr(args, "year_col", None) and args.year_col in df.columns: cands.append(args.year_col)
    except Exception:
        pass
    for c in ("Year","year","release_date","ReleaseDate","releaseDate"):
        try:
            if c in df.columns and c not in cands: cands.append(c)
        except Exception:
            continue
    for c in cands:
        y = _extract_year_from_value_pre(row.get(c))
        if y is not None: return y
    return None

# ===== Prepass PEGI detection =====

def _pegi_any_number_or_none(val) -> Optional[int]:
    if val is None: return None
    m = re.search(r"(\d{1,2})", str(val))
    if m:
        try: return int(m.group(1))
        except Exception: return None
    return None

# Case-insensitive column resolver

def _resolve_col_name(df: pd.DataFrame, desired: str) -> str:
    want = (desired or "").strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == want:
            return c
    return desired


def _global_prepass_fill(df: pd.DataFrame, args, print_lock: Optional[threading.Lock]):
    pre1994_set = 0; pegi_set = 0
    pegi_numeric = 0

    in_col = getattr(args, "pegi_input_col", "PEGI_Rating")
    if in_col not in df.columns:
        in_col = _resolve_col_name(df, in_col)
        args.pegi_input_col = in_col

    for idx, row in df.iterrows():
        plat = str(row.get(args.platform_col, "")).strip()
        y = _extract_year_from_row_any(df, row, args)
        # Historical rule
        if (y is not None) and (y <= 1994) and (not _platform_is_sega_vrc(plat)) and (not _platform_is_3do(plat)):
            df.at[idx, "new_rating"] = 3
            df.at[idx, "new_rating_source"] = "pre-1994"
            pre1994_set += 1
            continue
        # PEGI numeric present in input CSV
        val = row.get(in_col)
        p = _pegi_any_number_or_none(val)
        if p is not None:
            pegi_numeric += 1
            if _is_empty_or_nan(row.get("new_rating")):
                df.at[idx, "new_rating"] = int(p)
                df.at[idx, "new_rating_source"] = "PEGI present"
                pegi_set += 1

    _print_section("PREPASS", True, print_lock)
    _print_kv({
        "pegi_input_col_resolved": in_col,
        "pegi_numeric_in_input": pegi_numeric,
    }, True, lock=print_lock)
    _print_kv({"pre1994_set": pre1994_set, "pegi_set": pegi_set}, True, lock=print_lock)
    return pre1994_set, pegi_set

# ===== Rate-aware, batched collection =====

def collect_first_rating_tiered(title: str, plat: str, year: Optional[int], threshold: int,
                                igdb: Optional[IGDBClient], wd_user_agent: str, wd_cache, rl_global: RPSLimiter, rl_wd: RPSLimiter,
                                gb: Optional[GiantBombClient], rawg: Optional[RAWGClient],
                                board_priority: List[str], verbose: bool=False,
                                print_lock: Optional[threading.Lock]=None,
                                api_executor: Optional[ThreadPoolExecutor]=None,
                                api_counter: Optional[APICallCounter]=None,
                                igdb_prefetch: Optional[IGDBPrefetchResult]=None) -> Dict[str, Any]:
    enriched: Dict[str, Any] = {}

    def _run_api(func):
        if not api_executor:
            return func()
        return api_executor.submit(func).result()

    def _has_priority_hit() -> bool:
        for b in board_priority:
            col = BOARD_COL[b]
            if enriched.get(col):
                _print_kv(
                    {col: enriched[col], f"{col}_source": enriched.get(f"{col}_source", "(unknown)")},
                    verbose,
                    title=f"[HIT] {b.upper()} — stopping",
                    lock=print_lock,
                )
                return True
        return False

    def _needs_any_boards(candidates: List[str]) -> bool:
        for b in candidates:
            col = BOARD_COL.get(b)
            if col and not enriched.get(col):
                return True
        return False

    # 1) IGDB (primary)
    if igdb:
        prefetched_game: Optional[Dict[str, Any]] = None
        prefetched_attempted = False
        prefetched_via_batch = False
        if igdb_prefetch and igdb_prefetch.fulfilled:
            prefetched_attempted = True
            prefetched_via_batch = igdb_prefetch.via_batch
            if igdb_prefetch.game:
                prefetched_game = igdb_prefetch.game

        payload: Dict[str, Any] = {}
        if prefetched_attempted:
            if prefetched_game:
                try:
                    payload = ratings_from_igdb_full(prefetched_game, igdb) or {}
                except Exception as e:
                    log.warning("IGDB prefetch parsing failed: %s", e)
                    payload = {}
        else:
            def _igdb_task():
                try:
                    game = igdb.search_game_tiered(title, plat, year, threshold)
                    if game:
                        return ratings_from_igdb_full(game, igdb) or {}
                except Exception as e:
                    log.warning("IGDB fetch failed: %s", e)
                return {}

            payload = _run_api(_igdb_task)

        if isinstance(payload, dict) and payload:
            enriched.update(payload)
            api_label = "IGDB-batch" if prefetched_via_batch else "IGDB"
            status = "HAVE AGE_RATINGS"
            if prefetched_via_batch:
                status += " (prefetch)"
            _progress_line(plat, title, api_label, status, verbose, print_lock)
            if _has_priority_hit():
                return enriched

    # 2) RAWG (ESRB only)
    if rawg and ("esrb" in board_priority) and _needs_any_boards(["esrb"]):
        def _rawg_task():
            try:
                results = rawg.search(title, year)
                best = None; best_s = -999
                for r0 in results:
                    s = 0
                    if (r0.get("name", "")) and (r0.get("name").strip().lower() == title.lower()):
                        s += 2
                    try:
                        y0 = int((r0.get("released") or "")[:4])
                        if year and abs(y0 - int(year)) <= 1:
                            s += 1
                    except Exception:
                        pass
                    if s > best_s:
                        best, best_s = r0, s
                if best is not None:
                    det = rawg.game_details(best.get("slug") or best.get("id"))
                    return ratings_from_rawg(det, plat) or {}
            except Exception as e:
                log.warning("RAWG fetch failed: %s", e)
            return {}

        payload = _run_api(_rawg_task)
        if isinstance(payload, dict) and payload:
            enriched.update(payload)
            status = "ESRB"
            if payload.get("rating_esrb"):
                status = f"ESRB={payload.get('rating_esrb')}"
            _progress_line(plat, title, "RAWG", status, verbose, print_lock)
            if _has_priority_hit():
                return enriched

    # 3) Wikidata (only PEGI/ESRB/CERO/USK)
    if _needs_any_boards(["pegi", "esrb", "cero", "usk"]):
        def _wd_task():
            try:
                wd_all = wikidata_ratings(title, plat, year, wd_user_agent, wd_cache, rl_global, rl_wd, verbose, api_counter=api_counter)
                if wd_all:
                    wd_map: Dict[str, Any] = {}
                    if wd_all.get("PEGI"):
                        wd_map.setdefault("rating_pegi", wd_all["PEGI"])
                        wd_map.setdefault("rating_pegi_source", "Wikidata")
                    if wd_all.get("ESRB"):
                        wd_map.setdefault("rating_esrb", wd_all["ESRB"])
                        wd_map.setdefault("rating_esrb_source", "Wikidata")
                    if wd_all.get("CERO"):
                        wd_map.setdefault("rating_cero", wd_all["CERO"])
                        wd_map.setdefault("rating_cero_source", "Wikidata")
                    if wd_all.get("USK"):
                        wd_map.setdefault("rating_usk", wd_all["USK"])
                        wd_map.setdefault("rating_usk_source", "Wikidata")
                    return wd_map
            except Exception as e:
                log.warning("Wikidata fetch failed: %s", e)
            return {}

        payload = _run_api(_wd_task)
        if isinstance(payload, dict) and payload:
            for k, v in payload.items():
                enriched.setdefault(k, v)
            _progress_line(plat, title, "Wikidata", "CANDIDATES", verbose, print_lock)
            if _has_priority_hit():
                return enriched

    # 4) GiantBomb (last resort)
    if gb and _needs_any_boards(board_priority):
        def _gb_task():
            try:
                results = gb.search_game(title, year)
                best = None; best_s = -1
                for r0 in results:
                    s = 0
                    if (r0.get("name", "")) and (r0.get("name").strip().lower() == title.lower()):
                        s += 2
                    try:
                        y0 = int((r0.get("original_release_date") or "")[:4])
                        if year and abs(y0 - int(year)) <= 1:
                            s += 1
                    except Exception:
                        pass
                    if s > best_s:
                        best, best_s = r0, s
                if best and best.get("api_detail_url"):
                    det = gb.game_details(best["api_detail_url"]) or {}
                    return parse_giantbomb_ratings(det.get("original_game_rating"))
            except Exception as e:
                log.warning("GiantBomb fetch failed: %s", e)
            return {}

        payload = _run_api(_gb_task)
        if isinstance(payload, dict) and payload:
            enriched.update(payload)
            for b in board_priority:
                col = BOARD_COL[b]
                if enriched.get(col):
                    _progress_line(plat, title, "GiantBomb", f"SUCCESS: {enriched[col]}", verbose, print_lock)
                    _print_kv(
                        {col: enriched[col], f"{col}_source": enriched.get(f"{col}_source", "GiantBomb")},
                        verbose,
                        title=f"[HIT] {b.upper()} from GiantBomb — stopping",
                        lock=print_lock,
                    )
                    break

    return enriched

# ===== Main =====

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--config", dest="config_path", required=True)
    ap.add_argument("--platforms-csv", dest="platforms_csv", required=True)
    ap.add_argument("--title-col", default="Name")
    ap.add_argument("--platform-col", default="Platform")
    ap.add_argument("--year-col", default=None)
    ap.add_argument("--per-platform-limit", type=int, default=0)
    ap.add_argument("--priority-col", type=str, default="")
    ap.add_argument("--descending", action="store_true")
    ap.add_argument("--threshold", type=int, default=78)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pegi-input-col", default="PEGI_Rating",
                    help="If present and non-empty, treat as final PEGI rating and skip all API calls for that row.")
    # Backfills
    ap.add_argument("--enable-giantbomb-backfill", action="store_true")
    ap.add_argument("--enable-rawg-backfill", action="store_true")
    # Verbose
    ap.add_argument("--verbose", action="store_true")
    # Summaries
    ap.add_argument("--summary-every", type=int, default=50)
    ap.add_argument("--summary-json", default="ratings_summary.json")
    ap.add_argument("--summary-platform-csv", default="ratings_summary_by_platform.csv")
    ap.add_argument("--progress-interval", type=int, default=0,
                    help="If >0, emit a progress log every N attempted rows.")
    # Rate limits
    ap.add_argument("--rps-global", type=float, default=3.0)
    ap.add_argument("--rps-igdb", type=float, default=3.8)
    ap.add_argument("--rps-wikidata", type=float, default=1.0)
    ap.add_argument("--rps-giantbomb", type=float, default=0.05)  # ≈ 200/hour
    ap.add_argument("--rps-rawg", type=float, default=1.5)
    ap.add_argument("--gb-hourly", type=int, default=190,
                    help="Max GiantBomb requests per hour per resource")
    ap.add_argument("--gb-min-spacing", type=float, default=20.0,
                    help="Minimum seconds between GiantBomb calls (velocity)")
    # Tiered boards
    ap.add_argument("--board-priority", default="pegi,esrb,cero,usk,3do,sega_vrc")
    # Concurrency
    ap.add_argument("--workers", type=int, default=6, help="Number of threads to process rows")
    ap.add_argument("--igdb-batch-size", type=int, default=0,
                    help="If >1, queue unresolved rows and issue IGDB multiqueries of up to this size.")
    ap.add_argument("--igdb-batch-wait", type=float, default=0.05,
                    help="Seconds to wait for other rows before flushing a partial IGDB batch (when enabled).")
    # Resume / final-result cache
    ap.add_argument("--resume-from-cache", dest="resume_from_cache", action="store_true", default=True,
                    help="Reuse final ratings from SQLite for same (title, platform, year) before any API calls (default ON).")
    ap.add_argument("--no-resume-from-cache", dest="resume_from_cache", action="store_false")
    # Cross-title reuse
    ap.add_argument("--share-title-results", dest="share_title_results", action="store_true",
                    help="Reuse previously fetched ratings for the same normalized title before issuing new API calls.")
    ap.add_argument("--no-share-title-results", dest="share_title_results", action="store_false",
                    help="Disable cross-title reuse cache.")
    ap.set_defaults(share_title_results=True)
    # Checkpointing
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="If >0, write CSV to disk every N attempted rows to preserve progress mid-run.")

    args = ap.parse_args()

    # Locks & trackers
    print_lock = threading.Lock()
    stats_lock = threading.Lock()
    df_lock = threading.Lock()
    api_counter = APICallCounter()

    if args.verbose:
        _print_kv({
            "Workers": args.workers,
            "GB hourly cap/resource": args.gb_hourly,
            "GB min spacing (s)": args.gb_min_spacing,
            "GB rps": args.rps_giantbomb,
            "IGDB rps": args.rps_igdb,
            "IGDB batch size": args.igdb_batch_size or "—",
            "IGDB batch wait (s)": args.igdb_batch_wait if args.igdb_batch_size > 1 else "—",
            "Resume from cache": args.resume_from_cache,
            "Checkpoint every": args.checkpoint_every or "—"
        }, True, title="Concurrency & Limits", lock=print_lock)

    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Limiters
    rl_global   = RPSLimiter(args.rps_global)
    rl_igdb     = RPSLimiter(args.rps_igdb)
    rl_wd       = RPSLimiter(args.rps_wikidata)
    rl_gb_rps   = RPSLimiter(args.rps_giantbomb)
    rl_rawg     = RPSLimiter(args.rps_rawg)
    gb_hour_search  = HourlyQuotaLimiter(args.gb_hourly, 3600)
    gb_hour_details = HourlyQuotaLimiter(args.gb_hourly, 3600)
    gb_spacing      = MinSpacingLimiter(args.gb_min_spacing)

    cache_conn = get_db()

    # Clients
    igdb = None
    if cfg.get("igdb", {}).get("client_id") and cfg.get("igdb", {}).get("client_secret"):
        igdb = IGDBClient(
            cfg["igdb"]["client_id"], cfg["igdb"]["client_secret"], cache_conn,
            rl_global, rl_igdb, print_lock=print_lock, verbose=args.verbose, api_counter=api_counter
        )

    igdb_batcher: Optional[IGDBBatcher] = None
    if igdb and args.igdb_batch_size and args.igdb_batch_size > 1:
        igdb_batcher = IGDBBatcher(igdb, args.igdb_batch_size, args.igdb_batch_wait)

    wd_user_agent = cfg.get("wikidata", {}).get("user_agent", "RatingsPipeline/1.0 (+contact)")

    gb = None
    if args.enable_giantbomb_backfill and cfg.get("giantbomb", {}).get("api_key"):
        gb = GiantBombClient(
            cfg["giantbomb"]["api_key"], cfg.get("giantbomb", {}).get("user_agent"), cache_conn,
            rl_global, rl_gb_rps, gb_hour_search, gb_hour_details, gb_spacing,
            print_lock=print_lock, verbose=args.verbose, api_counter=api_counter
        )

    rawg = None
    if args.enable_rawg_backfill and cfg.get("rawg", {}).get("api_key"):
        rawg = RAWGClient(cfg["rawg"]["api_key"], cache_conn, rl_global, rl_rawg, api_counter=api_counter)

    # Load data & select work set
    _print_section("LOAD & SELECT WORK", args.verbose, print_lock)
    plats_df = pd.read_csv(args.platforms_csv, header=0)
    plat_colname = plats_df.columns[0]
    platform_order = [str(x).strip() for x in plats_df[plat_colname].dropna().tolist() if str(x).strip()]
    allowed_platforms = set(x.lower() for x in platform_order)

    df = pd.read_csv(args.in_path)

    # Resolve column names (case-insensitive)
    args.title_col    = _resolve_col_name(df, args.title_col)
    args.platform_col = _resolve_col_name(df, args.platform_col)
    if args.year_col:
        args.year_col = _resolve_col_name(df, args.year_col)
    args.pegi_input_col = _resolve_col_name(df, getattr(args, "pegi_input_col", "PEGI_Rating"))

    _ = _global_prepass_fill(df, args, print_lock)

    work_indices: List[int] = []
    for p in platform_order:
        mask = df[args.platform_col].astype(str).str.lower() == p.lower()
        chunk = df[mask].copy()
        if args.priority_col and args.priority_col in chunk.columns:
            chunk = chunk.sort_values(by=args.priority_col, ascending=not args.descending)
        k = args.per_platform_limit if args.per_platform_limit and args.per_platform_limit > 0 else len(chunk)
        chosen = chunk.head(k)
        work_indices.extend(chosen.index.tolist())
        if args.verbose:
            _print_kv({"platform": p, "selected": len(chosen), "in-platform-total": len(chunk)}, True, lock=print_lock)

    if args.limit and args.limit > 0:
        work_indices = work_indices[:args.limit]
        if args.verbose:
            _print_kv({"global_cap_rows": len(work_indices)}, True, lock=print_lock)

    # Ensure output columns exist
    for c in ["new_rating","new_rating_source"]:
        if c not in df.columns: df[c] = ""

    # Platform policy (optional, from rightmost column)
    policy_map: Dict[str,str] = {}
    policy_map_canon: Dict[str,str] = {}
    try:
        if plats_df.shape[1] >= 2:
            _rightmost = plats_df.columns[-1]
            _plat_col = args.platform_col if args.platform_col in plats_df.columns else plats_df.columns[0]
            for _i, _r in plats_df.iterrows():
                _p_raw = str(_r.get(_plat_col, "")).strip()
                _v = str(_r.get(_rightmost, "")).strip()
                if _p_raw and _v:
                    policy_map[_p_raw] = _v
                    policy_map_canon[(str(_p_raw).lower())] = _v
    except Exception as _e:
        _print_section(f"WARNING: failed to read platforms.csv policy: {type(_e).__name__}: {_e}", args.verbose, print_lock)

    # Summaries
    stats = {
        "rows_selected": len(work_indices),
        "rows_attempted": 0,
        "rows_with_any_rating": 0,
        "cache_negative_hits": 0,
        "source_counts": {"InputCSV":0, "IGDB":0, "Wikidata":0, "GiantBomb":0, "RAWG":0},
        "board_counts": {"PEGI":0,"ESRB":0,"CERO":0,"USK":0,"3DO":0,"SEGA_VRC":0},
        "start_time": time.time(),
    }

    progress = ProgressReporter(
        total=len(work_indices) or 1,
        row_interval=args.progress_interval or 0,
        lock=print_lock,
    )

    per_platform = defaultdict(lambda: {"selected":0,"attempted":0,"with_any_rating":0,"cached_negative":0})
    for idx in work_indices:
        p = str(df.loc[idx, args.platform_col]).strip()
        per_platform[p]["selected"] += 1

    board_priority = [b.strip().lower() for b in args.board_priority.split(",") if b.strip() and b.strip().lower() in BOARD_COL]

    api_executor: Optional[ThreadPoolExecutor] = None
    parallel_sources = 1  # Wikidata always runs
    if igdb:
        parallel_sources += 1
    if rawg and ("esrb" in board_priority):
        parallel_sources += 1
    if parallel_sources > 1:
        api_workers = min(16, max(4, args.workers * parallel_sources))
        api_executor = ThreadPoolExecutor(max_workers=api_workers)

    inflight = InFlightDeduper()

    title_share_lock = threading.Lock()
    title_share_state: Dict[str, Dict[str, Any]] = {}

    def _share_platform_key(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        base = _canonical_platform_name(name)
        key = (base or name).strip().lower()
        return key or None

    def _share_year_value(year_val: Optional[int]) -> Optional[int]:
        if year_val is None:
            return None
        try:
            return int(year_val)
        except Exception:
            try:
                return int(float(str(year_val)))
            except Exception:
                return None

    def remember_title_result(norm_title_key: str, payload: Dict[str, Any], year: Optional[int], platform: Optional[str]) -> None:
        if not args.share_title_results or not norm_title_key or not payload:
            return
        filtered = {k: v for k, v in payload.items() if v not in (None, "", [], {}, "None")}
        if not filtered:
            return
        boards_present = {b for b, col in BOARD_COL.items() if filtered.get(col)}
        if not boards_present:
            return
        platform_specific = any(b in {"3do", "sega_vrc"} for b in boards_present)
        platform_key = _share_platform_key(platform) if platform_specific else "__ANY__"
        year_key = _share_year_value(year)
        with title_share_lock:
            entry = title_share_state.get(norm_title_key)
            if entry is None:
                entry = {
                    "payload": dict(filtered),
                    "years": set(),
                    "platforms": set(),
                }
                title_share_state[norm_title_key] = entry
            else:
                existing_payload = entry["payload"]
                for k, v in filtered.items():
                    if not existing_payload.get(k):
                        existing_payload[k] = v
            if year_key is None:
                entry["years"].add(None)
            else:
                entry["years"].add(year_key)
            if platform_key:
                entry["platforms"].add(platform_key)

    def lookup_title_result(norm_title_key: str, year: Optional[int], platform: Optional[str], boards: List[str]) -> Optional[Dict[str, Any]]:
        if not args.share_title_results or not norm_title_key:
            return None
        with title_share_lock:
            entry = title_share_state.get(norm_title_key)
            if not entry:
                return None
            payload = entry.get("payload") or {}
            if not payload:
                return None
            if boards:
                if not any(payload.get(BOARD_COL[b]) for b in boards if b in BOARD_COL):
                    return None
            years = entry.get("years", set())
            if year is not None and years:
                year_int = _share_year_value(year)
                if year_int is not None:
                    ok = False
                    for y in years:
                        if y is None:
                            ok = True
                            break
                        try:
                            if abs(year_int - int(y)) <= 1:
                                ok = True
                                break
                        except Exception:
                            continue
                    if not ok:
                        return None
            platforms = entry.get("platforms", set())
            if platforms and "__ANY__" not in platforms:
                plat_key = _share_platform_key(platform)
                if not plat_key or plat_key not in platforms:
                    return None
            return dict(payload)

    # Worker
    def process_row(idx: int, n: int, total: int):
        dedupe_owner = False
        dedupe_key: Optional[str] = None
        dedupe_event: Optional[threading.Event] = None

        def _release_dedupe():
            if dedupe_owner and dedupe_key:
                inflight.release(dedupe_key)

        def _increment_attempted_locked():
            stats["rows_attempted"] += 1
            per_platform[plat]["attempted"] += 1

        def _emit_progress_locked():
            progress.maybe_emit(
                stats["rows_attempted"],
                stats["rows_with_any_rating"],
                dict(stats["source_counts"]),
                api_counter.snapshot(),
            )

        try:
            row = df.loc[idx]
            title = str(row.get(args.title_col, "")).strip()
            plat  = str(row.get(args.platform_col, "")).strip()
            if not title or not plat or plat.lower() not in allowed_platforms:
                return None

            if args.verbose:
                _progress_line(plat, title, "ROW", f"start [{n}/{total}]", True, print_lock)

            norm_title_key = norm(title)

            is_3do = platform_name_matches(["3DO"], plat)
            is_sega_vrc = platform_name_matches([
                "Master System","Sega Master System","Genesis","Mega Drive","Sega Genesis",
                "Game Gear","Sega Game Gear","Sega CD","Mega-CD","Mega CD","32X","Sega 32X"
            ], plat)
            board_priority_local = [
                b for b in board_priority
                if not ((b == "3do" and not is_3do) or (b == "sega_vrc" and not is_sega_vrc))
            ]

            year: Optional[int] = None
            if args.year_col and args.year_col in df.columns:
                try:
                    year = int(float(str(row.get(args.year_col)).strip()))
                except Exception:
                    year = _extract_year_from_row_any(df, row, args)
            else:
                year = _extract_year_from_row_any(df, row, args)

            def _apply_cached_result(cached_final: Dict[str, Any], source_label: str = "Cache") -> None:
                def _pick_best_from_cached(d: Dict[str,Any]) -> Tuple[Optional[str], Optional[str]]:
                    pri = ['pegi','esrb','cero','usk','3do','sega_vrc','vrc']
                    for b in pri:
                        rv = d.get(f'rating_{b}') or d.get(f'{b}_rating')
                        if rv not in (None, '', 'None'):
                            return str(rv), b.upper()
                    return None, None

                _nr, _src = _pick_best_from_cached(cached_final)
                if _nr is not None:
                    try:
                        df.at[idx, 'new_rating'] = _nr
                        df.at[idx, 'new_rating_source'] = (_src + ' (cache)') if _src else 'cache'
                    except Exception:
                        pass
                with df_lock:
                    for k, v in cached_final.items():
                        if k in df.columns:
                            df.at[idx, k] = v
                        else:
                            df[k] = ""
                            df.at[idx, k] = v
                with stats_lock:
                    _increment_attempted_locked()
                    stats["rows_with_any_rating"] += 1
                    per_platform[plat]["with_any_rating"] += 1
                    for s in ["InputCSV","IGDB","Wikidata","GiantBomb","RAWG"]:
                        if any((k.endswith("_source") and cached_final.get(k) == s) for k in cached_final):
                            stats["source_counts"][s] += 1
                            break
                    for b in board_priority:
                        c = BOARD_COL[b]
                        if cached_final.get(c):
                            stats["board_counts"][("SEGA_VRC" if b=="sega_vrc" else ("3DO" if b=="3do" else b.upper()))] += 1
                            break
                    if args.checkpoint_every and (stats["rows_attempted"] % args.checkpoint_every == 0):
                        with df_lock:
                            df.to_csv(args.out_path, index=False)
                    _emit_progress_locked()
                remember_title_result(norm_title_key, cached_final, year, plat)
                _progress_line(plat, title, source_label, "RESUMED", args.verbose, print_lock)

            # If input CSV already has a PEGI rating, treat it as final and skip APIs
            pegi_int = _pegi_any_number_or_none(row.get(args.pegi_input_col)) if args.pegi_input_col in df.columns else None
            if pegi_int is not None:
                with df_lock:
                    df.at[idx, "new_rating"] = int(pegi_int)
                    df.at[idx, "new_rating_source"] = "PEGI present"
                with stats_lock:
                    _increment_attempted_locked()
                    stats["rows_with_any_rating"] += 1
                    per_platform[plat]["with_any_rating"] += 1
                    stats["source_counts"]["InputCSV"] += 1
                    stats["board_counts"]["PEGI"] += 1
                    if args.checkpoint_every and (stats["rows_attempted"] % args.checkpoint_every == 0):
                        with df_lock:
                            df.to_csv(args.out_path, index=False)
                    _emit_progress_locked()
                remember_title_result(
                    norm_title_key,
                    {"rating_pegi": f"PEGI {int(pegi_int)}", "rating_pegi_source": "InputCSV"},
                    year,
                    plat,
                )
                _progress_line(plat, title, "InputCSV", f"SKIPPED APIS (PEGI present): PEGI {pegi_int}", args.verbose, print_lock)
                return None

            # Already has any rating in CSV? (skip APIs)
            if not _is_empty_or_nan(row.get("new_rating", "")):
                with stats_lock:
                    _increment_attempted_locked()
                    if args.checkpoint_every and (stats["rows_attempted"] % args.checkpoint_every == 0):
                        with df_lock:
                            df.to_csv(args.out_path, index=False)
                    _emit_progress_locked()
                _progress_line(plat, title, "InputCSV", "SKIPPED APIS (rating columns already filled)", args.verbose, print_lock)
                return None

            # Policy-based short-circuit (e.g., "3 if release_date<1994 else pegi>…")
            policy = (policy_map.get(plat) or policy_map_canon.get(plat.lower()) or "").lower()
            def _to_int_year(y):
                try:
                    return int(float(str(y).strip()))
                except Exception:
                    return None
            y2 = _to_int_year(row.get(args.year_col, "")) if args.year_col else year
            if ("3 if" in policy) and ("release_date<1994" in policy):
                if (y2 is None) or (y2 <= 1994):
                    with df_lock:
                        df.at[idx, "new_rating"] = 3
                        df.at[idx, "new_rating_source"] = "pre-1994"
                    with stats_lock:
                        _increment_attempted_locked()
                        stats["rows_with_any_rating"] += 1
                        per_platform[plat]["with_any_rating"] += 1
                        stats["board_counts"]["PEGI"] += 1
                        if args.checkpoint_every and (stats["rows_attempted"] % args.checkpoint_every == 0):
                            with df_lock:
                                df.to_csv(args.out_path, index=False)
                        _emit_progress_locked()
                    remember_title_result(
                        norm_title_key,
                        {"rating_pegi": "PEGI 3", "rating_pegi_source": "Policy"},
                        year,
                        plat,
                    )
                    _progress_line(plat, title, "Policy", 'PRE-1994 RULE: new_rating=3', args.verbose, print_lock)
                    return None

            # Resume-from-cache first
            if args.resume_from_cache:
                cached_final = final_cache_get(cache_conn, title, plat, year)
                if cached_final:
                    _apply_cached_result(cached_final)
                    return None

            shared_payload = lookup_title_result(norm_title_key, year, plat, board_priority_local)
            if shared_payload:
                _apply_cached_result(shared_payload, source_label="TitleShare")
                return None

            if args.resume_from_cache:
                cached_negative = final_cache_get_negative(cache_conn, title, plat, year)
                if cached_negative:
                    source_label = str(cached_negative.get("__source", "cache: no rating"))
                    with df_lock:
                        if "new_rating_source" not in df.columns:
                            df["new_rating_source"] = ""
                        df.at[idx, "new_rating_source"] = source_label
                    with stats_lock:
                        _increment_attempted_locked()
                        stats["cache_negative_hits"] += 1
                        per_platform[plat]["cached_negative"] += 1
                        if args.checkpoint_every and (stats["rows_attempted"] % args.checkpoint_every == 0):
                            with df_lock:
                                df.to_csv(args.out_path, index=False)
                        _emit_progress_locked()
                    _progress_line(plat, title, "Cache", "NEGATIVE (cached miss)", args.verbose, print_lock)
                    return None

            if args.resume_from_cache:
                dedupe_key = _final_key(title, plat, year)
                dedupe_owner, dedupe_event = inflight.acquire(dedupe_key)
                if not dedupe_owner and dedupe_event is not None:
                    _progress_line(plat, title, "Cache", "WAITING FOR IN-FLIGHT RESULT", args.verbose, print_lock)
                    dedupe_event.wait()
                    cached_final = final_cache_get(cache_conn, title, plat, year)
                    if cached_final:
                        _apply_cached_result(cached_final)
                    else:
                        with stats_lock:
                            _increment_attempted_locked()
                            if args.checkpoint_every and (stats["rows_attempted"] % args.checkpoint_every == 0):
                                with df_lock:
                                    df.to_csv(args.out_path, index=False)
                            _emit_progress_locked()
                    _progress_line(plat, title, "Cache", "WAIT COMPLETE (no cached result)", args.verbose, print_lock)
                    return None

            igdb_prefetch_result: Optional[IGDBPrefetchResult] = None
            if igdb_batcher:
                try:
                    igdb_prefetch_result = igdb_batcher.fetch(title, plat, year, args.threshold)
                except Exception as _batch_e:
                    log.warning("IGDB batch prefetch failed: %s", _batch_e)
                    igdb_prefetch_result = IGDBPrefetchResult(fulfilled=False, via_batch=False, game=None)

            hit = collect_first_rating_tiered(
                title=title, plat=plat, year=year, threshold=args.threshold,
                igdb=igdb, wd_user_agent=wd_user_agent, wd_cache=cache_conn, rl_global=rl_global, rl_wd=rl_wd,
                gb=gb, rawg=rawg, board_priority=board_priority_local, verbose=args.verbose, print_lock=print_lock,
                api_executor=api_executor, api_counter=api_counter, igdb_prefetch=igdb_prefetch_result
            )

            with stats_lock:
                _increment_attempted_locked()

            if hit:
                # Determine which board/value we used
                used_board = None; used_value = None
                for b in board_priority_local:
                    c = BOARD_COL[b]
                    if hit.get(c):
                        used_board = b; used_value = hit.get(c)
                        break
                if used_board is None:
                    for b, c in BOARD_COL.items():
                        if hit.get(c):
                            used_board, used_value = b, hit.get(c)
                            break

                # Determine API source
                api_src = None
                for s in ["IGDB","Wikidata","GiantBomb","RAWG","InputCSV","Policy"]:
                    if any((k.endswith("_source") and hit.get(k) == s) for k in hit):
                        api_src = s
                        break

                age_int, board_label = _board_to_numeric(used_board or "", str(used_value or ""))
                with df_lock:
                    if age_int is not None:
                        df.at[idx, "new_rating"] = int(age_int)
                    if api_src and board_label:
                        df.at[idx, "new_rating_source"] = f"{api_src} ({board_label})"
                    elif api_src:
                        df.at[idx, "new_rating_source"] = api_src
                try:
                    final_cache_put(cache_conn, title, plat, year, hit)
                except Exception:
                    pass
                remember_title_result(norm_title_key, hit, year, plat)

                with stats_lock:
                    stats["rows_with_any_rating"] += 1
                    per_platform[plat]["with_any_rating"] += 1
                    src = None
                    for s in ["InputCSV","IGDB","Wikidata","GiantBomb","RAWG"]:
                        if any((k.endswith("_source") and hit.get(k) == s) for k in hit):
                            src = s
                            break
                    if src:
                        stats["source_counts"][src] += 1
                    for b in board_priority_local:
                        c = BOARD_COL[b]
                        if hit.get(c):
                            stats["board_counts"][("SEGA_VRC" if b=="sega_vrc" else ("3DO" if b=="3do" else b.upper()))] += 1
                            break
                    _emit_progress_locked()
            else:
                _progress_line(plat, title, "APIs", "NO RATING FOUND", args.verbose, print_lock)
                try:
                    final_cache_put_negative(cache_conn, title, plat, year)
                except Exception:
                    pass

            # periodic running summary + checkpoint
            with stats_lock:
                attempted = stats["rows_attempted"]
                if attempted and ((n % args.summary_every == 0) or (n == total)):
                    pct = 100.0 * stats["rows_with_any_rating"] / max(1, attempted)
                    log.info(
                        "Coverage so far: %d/%d attempted (%.1f%%) have ≥1 rating | InputCSV=%d, IGDB=%d, Wikidata=%d, GiantBomb=%d, RAWG=%d",
                        stats["rows_with_any_rating"], attempted, pct,
                        stats["source_counts"]["InputCSV"],
                        stats["source_counts"]["IGDB"],
                        stats["source_counts"]["Wikidata"],
                        stats["source_counts"]["GiantBomb"],
                        stats["source_counts"]["RAWG"]
                    )
                if args.checkpoint_every and (attempted % args.checkpoint_every == 0):
                    with df_lock:
                        df.to_csv(args.out_path, index=False)
                _emit_progress_locked()

            return None
        finally:
            _release_dedupe()

    # Run threaded
    _print_section("PROCESS (TIERED + MULTITHREADED)", args.verbose, print_lock)
    total = len(work_indices)
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(process_row, idx, i, total) for i, idx in enumerate(work_indices, 1)]
            for _f in as_completed(futures):
                try:
                    _f.result()
                except Exception as __e:
                    _print_section(f"WORKER ERROR: {type(__e).__name__}: {__e}", True, print_lock)
    finally:
        if api_executor:
            api_executor.shutdown(wait=True)

    # Finalize
    _print_section("FINALIZE", args.verbose, print_lock)
    df.to_csv(args.out_path, index=False)
    log.info("Wrote %s", args.out_path)

    attempted = max(1, stats["rows_attempted"])
    coverage_pct = 100.0 * stats["rows_with_any_rating"] / attempted
    summary_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows_selected": stats["rows_selected"],
        "rows_attempted": stats["rows_attempted"],
        "rows_with_any_rating": stats["rows_with_any_rating"],
        "cache_negative_hits": stats["cache_negative_hits"],
        "coverage_pct_attempted": round(coverage_pct, 2),
        "source_counts": stats["source_counts"],
        "board_counts": stats["board_counts"],
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    plat_rows = []
    for plat_label, nums in per_platform.items():
        sel = nums["selected"]
        att = nums.get("attempted", 0)
        got = nums["with_any_rating"]
        neg = nums.get("cached_negative", 0)
        denom = att if att else sel
        pct = (100.0 * got / denom) if denom else 0.0
        plat_rows.append(
            {
                "Platform": plat_label,
                "Selected": sel,
                "Attempted": att,
                "WithAnyRating": got,
                "CachedNegative": neg,
                "CoveragePct": round(pct, 2),
            }
        )
    pd.DataFrame(plat_rows).sort_values(by=["CoveragePct","Selected"], ascending=[False, False]).to_csv(args.summary_platform_csv, index=False)

    log.info(
        "FINAL: %d/%d attempted (%.1f%%) have ≥1 rating | IGDB=%d, Wikidata=%d, GiantBomb=%d, RAWG=%d, cache-negative=%d. Wrote %s and %s",
        stats["rows_with_any_rating"], stats["rows_attempted"], coverage_pct,
        stats["source_counts"]["IGDB"], stats["source_counts"]["Wikidata"],
        stats["source_counts"]["GiantBomb"], stats["source_counts"]["RAWG"], stats["cache_negative_hits"],
        args.summary_json, args.summary_platform_csv
    )
    api_totals = api_counter.snapshot()
    if api_totals:
        log.info("API totals: %s", ", ".join(f"{k}={v}" for k, v in sorted(api_totals.items())))

if __name__ == "__main__":
    main()
