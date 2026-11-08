"""Minimal FMP client: loads the key from .env, retries, and never leaks the key into logs."""
import os
import threading
import time
from pathlib import Path

import requests

BASE_URL = "https://financialmodelingprep.com/stable/"
ROOT = Path(__file__).resolve().parent
# FMP silently truncates price responses at this many rows, newest first.
PRICE_ROW_CAP = 5000
# Ultimate allows 3,000 calls a minute. Staying under it avoids the 429s that stall long pulls.
MAX_CALLS_PER_MINUTE = 2400


def load_api_key():
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("FMP_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("FMP_API_KEY not found in environment or .env")


class FMPClient:
    def __init__(self, retries=7, timeout=90):
        self.key = load_api_key()
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _wait_for_slot(self):
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + 60 / MAX_CALLS_PER_MINUTE
        time.sleep(slot - now)

    def get(self, endpoint, **params):
        last_error = None
        for attempt in range(self.retries):
            self._wait_for_slot()
            try:
                r = self.session.get(
                    BASE_URL + endpoint,
                    params={**params, "apikey": self.key},
                    timeout=self.timeout,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    last_error = f"HTTP {r.status_code}"
                    # Rate limits reset on a per-minute window, so short sleeps just burn retries.
                    time.sleep(min(5 * 2 ** attempt, 65))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"{endpoint} returned HTTP {r.status_code}")
                return r.json() if r.content else []
            except requests.RequestException as e:
                last_error = type(e).__name__
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{endpoint} failed after {self.retries} tries: {last_error}")

    def price_history(self, symbol, start, end, chunk_years=10):
        """Daily rows for a symbol, requested in chunks to stay under PRICE_ROW_CAP."""
        history = {}
        for year in range(int(start[:4]), int(end[:4]) + 1, chunk_years):
            lo = max(start, f"{year}-01-01")
            hi = min(end, f"{year + chunk_years - 1}-12-31")
            rows = self.get("historical-price-eod/light", symbol=symbol, **{"from": lo, "to": hi})
            if len(rows) >= PRICE_ROW_CAP:
                raise RuntimeError(f"{symbol} hit the price row cap for {lo}..{hi}")
            history.update({row["date"]: row for row in rows})
        return [history[date] for date in sorted(history)]
