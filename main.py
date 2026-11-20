"""GTA VI trophy-rarity forecast for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI (PS5), its earn rate on
Exophase at a case's target, as five quantiles (5/25/50/75/95%).

The program is run once per case in a sandbox with no network.  It reads
every ``*.in`` file in $MAKEFUN_INPUT_DIR (else /input/) and writes one
``<id>.out`` per case to $MAKEFUN_OUTPUT_DIR (else /output/).

Model
-----
Two regimes, picked from what the case's history actually shows.

1. Flat history (the practice cases).  Every read lists the same players
   tracked and the same rates: the page Exophase shows is stale, it has not
   refreshed since before the first read.  By the target a refresh reveals
   the players who joined in the meantime, and a trophy's rate is diluted:
   new players join far faster than they earn trophies in their first hours.
   For a trophy last at rate r (percent) with e earners among N players:

       rate(target) = r * (1 + (G - 1) * u) / G

   where G is how much the tracked-player count grows between the first read
   and the target (G = exp(0.35 * hours), launch-night growth, capped at
   12) and u is the share of the trophy's old earn rate that newcomers reach
   by the target.  u grows with how early/easy the trophy is, proxied by its
   own launch-night rate: u = 0.10 * sqrt(r / r_max).  The 100% trophy stays
   at 100% (the house's recording shows the one rate at 100% does not drop
   on a refresh).  The quantiles span 0.40x to 3.5x the median factor,
   because how fast newcomers earn is genuinely uncertain.

2. Moving history (the final).  Exophase has refreshed, so the reads trace
   real growth.  For each trophy, fit a line to logit(rate) against
   ln(1 + effective hours since release) over the distinct reads, where an
   effective hour counts weekend hours (Sat 00:00Z to Mon 00:00Z) at 1.2x --
   the final's window holds a full weekend, and players play more then.
   The slope is shrunk towards the median slope of the trophy's tier
   (platinum barely moves, bronze story trophies climb fast), the forecast
   is anchored on the last read's rate, and the slope is floored at 0 for
   horizons over a day: over a launch week, player growth slows while
   progression continues, so rates rise; a short dilution dip must not be
   extrapolated for four days.  Where the history shows the curve bending
   over -- five or more distinct reads and a concave quadratic fitting
   clearly better than the line -- the forecast uses the quadratic's
   increment instead, so a saturating story trophy is not extrapolated
   straight to 100%; the increment is still floored at 0 for long horizons.
   A tier cap stops runaway growth
   (platinum 5%, gold 20%, silver 50%, bronze 92% by the target week).
   The spread combines a random-walk term growing with the horizon
   (0.40 logit per root-day), the fit's residual and a refresh-lag term.

Output is always valid: any well-formed case gets a file, every trophy one
row, quantiles non-decreasing in [0, 100], deterministic, standard library
only, well inside the 8-second limit.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Tier priors: slope of logit(rate) per unit ln(1 + effective hours), and the
# highest rate the tier plausibly reaches in the target week.
TIER_SLOPE_PRIOR = {"platinum": 0.10, "gold": 0.30, "silver": 0.60, "bronze": 0.90}
TIER_CAP = {"platinum": 5.0, "gold": 20.0, "silver": 50.0, "bronze": 92.0}
# Fallback rate (percent) for a trophy with no usable read at all.
TIER_DEFAULT_RATE = {"platinum": 0.10, "gold": 0.30, "silver": 0.80, "bronze": 1.50}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(value):
    """ISO 8601 to an aware datetime; None when unparseable."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def effective_hours(start, end):
    """Hours from start to end with weekend hours (Sat 00:00Z-Mon 00:00Z) at 1.2x.

    Walked day by day; the spans here are days, so this is cheap.
    """
    if end <= start:
        return 0.0
    total = 0.0
    cur = start
    while cur < end:
        nxt_day = (cur + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        nxt = min(nxt_day, end)
        hours = (nxt - cur).total_seconds() / 3600.0
        weight = 1.2 if cur.weekday() in (5, 6) else 1.0
        total += hours * weight
        cur = nxt
    return total


def distinct_reads(history):
    """Collapse consecutive reads with an identical page (same players and
    rates) to one read each: Exophase refreshes in steps, and a flat stretch
    carries one observation, not one per 10-minute slot."""
    out = []
    prev_key = None
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        key = (read.get("players_tracked"),
               tuple(rates) if isinstance(rates, list) else None)
        if key == prev_key:
            continue
        prev_key = key
        out.append(read)
    return out


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, residual_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        )
    else:
        resid = 0.0
    return a, b, resid


def fit_quad(xs, ys):
    """Least squares y = a + b*x + c*x^2 via normal equations; None if
    degenerate or fewer than 5 points."""
    n = len(xs)
    if n < 5:
        return None
    s0, s1, s2, s3, s4 = float(n), 0.0, 0.0, 0.0, 0.0
    t0, t1, t2 = 0.0, 0.0, 0.0
    for x, y in zip(xs, ys):
        x2 = x * x
        s1 += x
        s2 += x2
        s3 += x2 * x
        s4 += x2 * x2
        t0 += y
        t1 += x * y
        t2 += x2 * y
    m = [[s0, s1, s2, t0], [s1, s2, s3, t1], [s2, s3, s4, t2]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for k in range(col, 4):
                m[r][k] -= f * m[col][k]
    c = m[2][3] / m[2][2]
    b = (m[1][3] - m[1][2] * c) / m[1][1]
    a = (m[0][3] - m[0][1] * b - m[0][2] * c) / m[0][0]
    sse = sum((y - (a + b * x + c * x * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, c, sse


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def forecast_flat_row(trophy, last_rate, reads, target_at, r_max):
    tier = trophy.get("tier") or "bronze"
    if last_rate is None:
        last_rate = TIER_DEFAULT_RATE.get(tier, 1.5)
    if last_rate >= 99.99:
        return [99.5, 99.9, 100, 100, 100]
    first_at = None
    if reads:
        first_at = parse_time(reads[0].get("at")) or parse_time(reads[0].get("slot"))
    span_h = 0.0
    if first_at is not None and target_at is not None:
        span_h = max(0.0, (target_at - first_at).total_seconds() / 3600.0)
    growth = min(12.0, max(1.5, math.exp(0.35 * span_h)))
    u_med = 0.10 * math.sqrt(max(0.0, last_rate) / r_max) if r_max > 0 else 0.10
    f_med = (1.0 + (growth - 1.0) * u_med) / growth
    # The band is asymmetric on purpose: pinball charges a 95% quantile
    # below the truth 19x what it charges one above it, and the mirror holds
    # at 5%, so the upper tail reaches far (newcomers may earn fast) while
    # the lower tail stops near pure dilution.
    factors = [max(0.06, f_med * 0.40), f_med * 0.80, f_med, f_med * 1.60,
               f_med * 3.5]
    # A refresh drops the rates, so no quantile sits above 95% of the last
    # read; cap every factor, then keep the row non-decreasing.
    factors = [min(0.95, f) for f in factors]
    out = []
    for f in factors:
        q = clamp_pct(last_rate * f)
        out.append(round4(q))
    # Keep the row non-decreasing after rounding.
    for j in range(1, 5):
        if out[j] < out[j - 1]:
            out[j] = out[j - 1]
    return out


def forecast_trend_row(trophy, last_rate, series, tier_median, release_at,
                       target_at, checkpoint_at, horizon_min):
    tier = trophy.get("tier") or "bronze"
    if last_rate is None:
        last_rate = TIER_DEFAULT_RATE.get(tier, 1.5)
    if last_rate >= 99.99:
        return [99.5, 99.9, 100, 100, 100]

    slope_prior = TIER_SLOPE_PRIOR.get(tier, 0.9)
    # The cap stops runaway growth for a trophy still below its tier's
    # ceiling; a trophy already above it is not capped at all.
    cap = TIER_CAP.get(tier, 92.0) if last_rate <= TIER_CAP.get(tier, 92.0) else 99.99

    slope = None
    resid = 0.0
    quad_inc = None
    if len(series) >= 3:
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        fit = fit_line(xs, ys)
        lin_sse = None
        if fit is not None:
            _, b_fit, resid = fit
            n = min(len(series), 10)
            prior = tier_median.get(tier, slope_prior)
            slope = (n * b_fit + 4.0 * prior) / (n + 4.0)
            lin_sse = sum((y - (fit[0] + fit[1] * x)) ** 2
                          for x, y in zip(xs, ys))
        qf = fit_quad(xs, ys) if len(series) >= 5 else None
        if qf is not None and lin_sse is not None:
            qa, qb, qc, qsse = qf
            if qsse < 0.9 * lin_sse and qc < 0.05:
                quad_inc = (qa, qb, qc)
    if slope is None:
        slope = tier_median.get(tier, slope_prior)
    slope = min(3.0, max(-0.5, slope))
    if horizon_min > 1440:
        slope = max(0.0, slope)

    if release_at is None or target_at is None or checkpoint_at is None:
        x_target = math.log(1.0 + max(0.0, horizon_min / 60.0))
        x_last = 0.0
    else:
        x_target = math.log(1.0 + effective_hours(release_at, target_at))
        x_last = math.log(1.0 + effective_hours(release_at, checkpoint_at))

    if quad_inc is not None:
        qa, qb, qc = quad_inc
        inc = (qa + qb * x_target + qc * x_target * x_target) - (
            qa + qb * x_last + qc * x_last * x_last)
        if horizon_min > 1440:
            inc = max(0.0, inc)
        med_logit = logit(last_rate) + inc
    else:
        med_logit = logit(last_rate) + slope * (x_target - x_last)
    med = min(cap, sigmoid(med_logit))
    med = max(0.01, med)

    days = max(0.0, horizon_min) / 1440.0
    sigma = math.sqrt((0.40 ** 2) * days + (0.6 * resid) ** 2 + 0.12 ** 2)
    med_lg = logit(med)
    out = []
    for z in Z:
        q = sigmoid(med_lg + z * sigma)
        out.append(round4(min(99.99, max(0.01, q))))
    for j in range(1, 5):
        if out[j] < out[j - 1]:
            out[j] = out[j - 1]
    return out


def build_series(case, trophies, reads, release_at, checkpoint_at=None):
    """Per-trophy (x, logit rate) series over the distinct reads, plus the
    median fitted slope per tier (the shrinkage prior).

    The first distinct read is dropped when the next one shows the stale
    page refreshing -- players tracked jumping 1.5x or more while rates
    mostly fall.  That launch-night page reflects an earlier, hardcore
    cohort, not the start of the curve the later reads trace, and a line
    through it understates the growth the final week actually shows."""
    series_reads = reads
    if len(reads) >= 2:
        first, second = reads[0], reads[1]
        n0 = first.get("players_tracked") or 0
        n1 = second.get("players_tracked") or 0
        if n0 and n1 and n1 >= 1.5 * n0:
            r0 = first.get("rates") or []
            r1 = second.get("rates") or []
            pairs = [(a, b) for a, b in zip(r0, r1)
                     if isinstance(a, (int, float)) and isinstance(b, (int, float))]
            if pairs and sum(1 for a, b in pairs if b < a) > len(pairs) // 2:
                series_reads = reads[1:]
    series = {}
    for i, trophy in enumerate(trophies):
        pts = []
        for read in series_reads:
            rates = read.get("rates") if isinstance(read, dict) else None
            if not isinstance(rates, list) or i >= len(rates):
                continue
            rate = rates[i]
            if rate is None:
                continue
            try:
                rate = float(rate)
            except (TypeError, ValueError):
                continue
            at = parse_time(read.get("at")) or parse_time(read.get("slot"))
            if at is None or release_at is None:
                continue
            eff = effective_hours(release_at, at)
            pts.append((math.log(1.0 + eff), logit(rate)))
        series[i] = pts
    slopes_by_tier = {}
    for i, trophy in enumerate(trophies):
        pts = series.get(i, [])
        if len(pts) >= 3:
            fit = fit_line([p[0] for p in pts], [p[1] for p in pts])
            if fit is not None:
                tier = trophy.get("tier") or "bronze"
                slopes_by_tier.setdefault(tier, []).append(fit[1])
    tier_median = {}
    for tier, vals in slopes_by_tier.items():
        m = median(vals)
        if m is not None:
            tier_median[tier] = m
    return series, tier_median


def process_case(case):
    """Forecast one case: build its per-trophy series, then one row each."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    reads = distinct_reads(history)
    # r_max for the flat regime: the highest non-100 rate in the last read.
    last_read = history[-1] if history else None
    rates = []
    if isinstance(last_read, dict) and isinstance(last_read.get("rates"), list):
        rates = [r for r in last_read["rates"]
                 if isinstance(r, (int, float)) and r < 99.99]
    r_max = max(rates) if rates else 2.0

    release_at = None
    game = case.get("game") or {}
    if isinstance(game, dict) and game.get("release_date"):
        release_at = parse_time(str(game["release_date"]) + "T00:00:00Z")
    if release_at is None:
        release_at = parse_time(case.get("at"))
    series, tier_median = build_series(case, trophies, reads, release_at,
                                       checkpoint_at=parse_time(case.get("at")))

    horizon_min = case.get("horizon_minutes") or 0
    try:
        horizon_min = float(horizon_min)
    except (TypeError, ValueError):
        horizon_min = 0.0
    target_at = parse_time(case.get("target")) or parse_time(case.get("at"))
    checkpoint_at = parse_time(case.get("at"))

    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            rs = read.get("rates") if isinstance(read, dict) else None
            if isinstance(rs, list) and i < len(rs) and rs[i] is not None:
                try:
                    rate = float(rs[i])
                    break
                except (TypeError, ValueError):
                    continue
        last_rates.append(rate)

    flat = len(reads) <= 1
    if flat and reads and checkpoint_at is not None:
        first_at = parse_time(reads[0].get("at")) or parse_time(reads[0].get("slot"))
        if first_at is not None and (checkpoint_at - first_at).total_seconds() > 24 * 3600:
            flat = False

    rows = []
    for i, trophy in enumerate(trophies):
        if flat:
            q = forecast_flat_row(trophy, last_rates[i], reads, target_at, r_max)
        else:
            q = forecast_trend_row(trophy, last_rates[i], series.get(i, []),
                                   tier_median, release_at, target_at,
                                   checkpoint_at, horizon_min)
        rows.append({"id": trophy.get("id"), "q": q})
    return {"trophies": rows}


def fallback_forecast(case):
    """A valid forecast for any case with a trophy list, even if the model
    path fails: tier defaults with a wide spread. Never raise for a
    well-formed case."""
    rows = []
    for trophy in case.get("trophies") or []:
        tier = (trophy or {}).get("tier") or "bronze"
        base = TIER_DEFAULT_RATE.get(tier, 1.5)
        lg = logit(base)
        rows.append({"id": (trophy or {}).get("id"), "q": [
            round4(sigmoid(lg + z * 0.8)) for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            forecast = process_case(case)
        except Exception:
            try:
                forecast = fallback_forecast(case)
            except Exception:
                continue
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
