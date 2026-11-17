"""GTA VI Twitch CCU forecast (gta6-twitch-ccu-forecast-01).

Forecasts hour-by-hour Twitch concurrent viewers for GTA VI's release day
(2026-11-19 UTC) from the first hour's print, as five quantiles per hour
(hours 2..24).

Method, per hour h = 2..24, fit at runtime from the case's comparables:
    log(viewers_h) = a_h + b_h * log(viewers_1)     (OLS in log space)
  - median = the fitted line at the case's own hour-1 print
  - spread = max(stdev(residuals), 1.4826 * MAD(residuals))
             * sqrt(1 + 1/n + (x0 - xbar)^2 / Sxx)   (prediction interval)
  - quantiles from a NormalDist at the case's levels

Training filter: only launches whose release day had already started by
hour 1 train the regression, i.e. whose day peak is at most 20x the hour-1
print. A launch with a peak above 20x hour 1 had not started at hour 1 (a
single global unlock later in the day, e.g. a PC game at 10:00/17:00 UTC,
whose hour-1 print is only the pre-release baseline); including it would
flatten the slope and, on a large hour-1 print, predict an absurd collapse.

Fallbacks: if fewer than 5 trained launches report an hour, use the median
log-ratio to hour 1 over all comparables; if that is empty too, persist
hour 1 with the house baseline's widening spread.

Standard library only. Deterministic. No network.
"""

import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [statistics.NormalDist().inv_cdf(t) for t in LEVELS]
PEAK_RATIO = 20.0
MIN_TRAIN = 5
BASE_SIGMA = 0.15
MAX_VIEWERS = 1_000_000_000.0


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp ('...Z') to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_hours(comp):
    """Return ({hour: viewers} for hours 1..24 of a comparable's release day, peak).

    `comp` is one comparables entry. Its `hourly` list holds one point per
    hour from 14 days before to 3 days after release; the release day starts
    at `day_start`. Only points with viewers > 0 are usable (logs need it).
    """
    day_start = parse_ts(comp["day_start"])
    hours = {}
    for row in comp.get("hourly") or []:
        v = row.get("viewers")
        if v is None or v <= 0:
            continue
        try:
            at = parse_ts(row["from"])
        except (KeyError, ValueError):
            continue
        off = (at - day_start).total_seconds() / 3600.0
        h = int(off) + 1
        if 1 <= h <= 24 and abs(off - int(off)) < 1e-9:
            hours[h] = float(v)
    peak = max(hours.values()) if hours else None
    return hours, peak


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, xbar, sxx, residuals)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, xbar, sxx, resid


def robust_spread(resid):
    """max(sample stdev, 1.4826 * MAD) of residuals; 0 when degenerate."""
    if len(resid) < 2:
        return 0.0
    sd = statistics.stdev(resid)
    med = statistics.median(resid)
    mad = statistics.median([abs(r - med) for r in resid])
    return max(sd, 1.4826 * mad)


def quantiles(median_log, spread):
    """Five quantiles at LEVELS from a lognormal with log-median and log-spread."""
    out = []
    for z in Z:
        q = math.exp(median_log + z * spread)
        q = min(max(q, 0.01), MAX_VIEWERS)
        out.append(q)
    # enforce non-decreasing (construction already gives it; guard float noise)
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_hour(h, p1, trained, all_comps):
    """Forecast one hour. Returns the five quantiles.

    trained: list of (hour1, {hour: viewers}) for filter-passing comparables.
    all_comps: list of (hour1, {hour: viewers}) for every comparable.
    """
    x0 = math.log(p1)
    pairs = [(math.log(h1), math.log(hours[h])) for h1, hours in trained
             if h in hours and h1 > 0]
    if len(pairs) >= MIN_TRAIN:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        a, b, xbar, sxx, resid = ols(xs, ys)
        median_log = a + b * x0
        base = robust_spread(resid)
        if sxx > 0:
            lev = math.sqrt(1.0 + 1.0 / len(xs) + (x0 - xbar) ** 2 / sxx)
        else:
            lev = math.sqrt(1.0 + 1.0 / len(xs))
        return quantiles(median_log, base * lev)
    ratios = [math.log(hours[h] / h1) for h1, hours in all_comps
              if h in hours and h1 > 0]
    if ratios:
        med = statistics.median(ratios)
        return quantiles(x0 + med, robust_spread(ratios))
    # nothing to learn from: persist hour 1, house-style widening spread
    return quantiles(x0, BASE_SIGMA * math.sqrt(h - 1))


def forecast_case(case):
    """Forecast all 23 hours of one case. Returns the output dict."""
    known = case.get("known") or []
    p1 = float(known[0].get("viewers", 1)) if known else 1.0
    if p1 < 1:
        p1 = 1.0
    trained = []
    all_comps = []
    for comp in case.get("comparables") or []:
        hours, peak = release_day_hours(comp)
        h1 = hours.get(1)
        if h1 is None or h1 <= 0:
            continue
        all_comps.append((h1, hours))
        if peak is not None and peak <= PEAK_RATIO * h1:
            trained.append((h1, hours))
    rows = []
    for h in range(2, 25):
        rows.append({"hour": h, "q": forecast_hour(h, p1, trained, all_comps)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        dest = OUTPUT_DIR / (path.stem + ".out")
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
