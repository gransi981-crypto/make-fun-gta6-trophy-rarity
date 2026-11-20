"""gta6-trophy-rarity-forecast-01 entry: forecast each GTA VI PS5 trophy's Exophase
earn rate at the case's target as five quantiles, scored by pinball loss on the
logit scale.

Model (standard library only, deterministic, offline):
  - An earn rate is a ratio: earners / players_tracked. Exophase refreshes a
    game's counts on its own schedule, hours apart, so the recorded history is a
    staircase: long flat stretches, then a jump in players_tracked that usually
    drags most rates DOWN (the newly tracked players hold fewer trophies than
    the launch-night sample), while earners keep growing as players progress.
  - Per trophy, fit a least-squares line of logit(rate) on ln(1 + hours since
    2026-11-19T00:00Z) over the distinct reads of the last 24 hours, like the
    house curve baseline -- then improve on it: shrink the slope toward
    persistence (no change) by how noisy the fit is, anchor the forecast at the
    last read's rate, and widen the spread where the trend is uncertain.
  - Spread (logit units) combines the starter's random-walk term, the trend
    fit's prediction error, the size of the refresh shocks seen in the history,
    and a wide allowance when the history is entirely flat (a refresh is then
    overdue: the practice cases are exactly that, and the first refresh moved
    the average rate about 1.5 logit units, which is why persistence scores
    ~1.51 there).
  - A history with no movement at all gets its median shifted down by 1 logit
    unit (dilution at the next refresh) except for a trophy already at 100%,
    which stays at 100%.

  - Quantiles are inv_logit(median_logit + z_tau * spread), z = the standard
    normal quantiles at 0.05/0.25/0.5/0.75/0.95, rounded to 4 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's 0.5).
SIGMA_RW = 0.5
# Launch-day origin for the log-time trend, matching the house curve baseline.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
# Median shift (logit units) when the history is entirely flat: the next
# Exophase refresh dilutes the rates. Calibrated on the practice refresh,
# where the average rate moved about 1.5 logit units.
FLAT_SHIFT = -1.0
# A trend may not move a trophy's median more than this (logit units) from
# its last rate: the future can break a trend, and a wild extrapolation
# pays at every quantile.
MAX_TREND_MOVE = 2.5
# Extra spread (logit units) when the history is entirely flat.
FLAT_SPREAD = 0.8
# Spread floor and cap (logit units): never claim more certainty than the
# data supports, never go so wide the quantiles pay at every level.
SPREAD_MIN = 0.35
SPREAD_MAX = 2.0


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_epoch(t: datetime) -> float:
    return (t - EPOCH).total_seconds() / 3600.0


def distinct_reads(history: list) -> list:
    """Collapse consecutive reads with identical players and rates: only a
    refresh changes anything, and identical reads carry no information."""
    out = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if key != prev:
            out.append(read)
            prev = key
    return out


def fit_trend(xs: list, ys: list):
    """Least-squares line y = a + b*x. Returns (a, b, s_res, se_pred_at)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    s_res = math.sqrt(ss_res / max(n - 2, 1)) if n > 2 else 0.0
    se_b = s_res / math.sqrt(sxx) if s_res > 0 else 0.0
    return a, b, s_res, se_b, mx, sxx, n


def refresh_shock_scale(states: list, n_trophies: int) -> float:
    """Typical size of a refresh shock, in logit units: for each pair of
    consecutive distinct reads, the median |change in logit(rate)| across
    trophies; the largest such median over the history."""
    best = 0.0
    for prev, cur in zip(states, states[1:]):
        diffs = []
        for i in range(n_trophies):
            r0 = prev["rates"][i]
            r1 = cur["rates"][i]
            if r0 is None or r1 is None:
                continue
            diffs.append(abs(logit(r1) - logit(r0)))
        if diffs:
            diffs.sort()
            med = diffs[len(diffs) // 2]
            best = max(best, med)
    return best


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(max(hours_since_epoch(target), -0.9))

    states = distinct_reads(case["history"])
    last = case["history"][-1]
    last_rates = last["rates"]
    # Last 24 hours of distinct reads (the house curve baseline's window).
    at = parse_time(case["at"])
    cutoff = at.timestamp() - 24 * 3600
    window = [r for r in states if parse_time(r["at"]).timestamp() >= cutoff]
    if len(window) < 3:
        window = states
    shock = refresh_shock_scale(states, n)
    rw = SIGMA_RW * math.sqrt(horizon_days)
    flat = len(states) <= 1

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            # Fall back to the trophy's last observed rate, else the
            # cross-trophy median of the last read.
            seen = [r["rates"][i] for r in reversed(case["history"])
                    if r["rates"][i] is not None]
            if seen:
                rate = seen[0]
            else:
                vals = sorted(v for v in last_rates if v is not None)
                rate = vals[len(vals) // 2] if vals else 1.0
        last_logit = logit(rate)

        if rate >= 99.99:
            # Everybody holds it; a bigger denominator cannot take it away.
            rows.append({"id": trophy["id"], "q": [99.5, 99.8, 100, 100, 100]})
            continue

        # Per-trophy trend on the window's distinct reads.
        xs, ys = [], []
        for read in window:
            r = read["rates"][i]
            if r is None:
                continue
            xs.append(math.log1p(max(hours_since_epoch(parse_time(read["at"])), -0.9)))
            ys.append(logit(r))
        median_logit = last_logit
        pred_var = 0.0
        if flat:
            median_logit = last_logit + FLAT_SHIFT
        elif len(xs) >= 3:
            fit = fit_trend(xs, ys)
            if fit is not None:
                _a, b, s_res, se_b, mx, sxx, m = fit
                if se_b > 0:
                    t_stat = abs(b) / se_b
                    w = t_stat * t_stat / (t_stat * t_stat + 3.0)
                else:
                    w = 0.9 if b != 0 else 0.0
                w = min(w, 0.9)
                x_last = xs[-1]
                move = w * b * (x_target - x_last)
                move = max(-MAX_TREND_MOVE, min(MAX_TREND_MOVE, move))
                median_logit = last_logit + move
                # Prediction error at the target, widened for the shrinkage.
                se_pred = s_res * math.sqrt(1.0 / m + (x_target - mx) ** 2 / sxx)
                pred_var = (se_pred * (1.0 + (1.0 - w))) ** 2
        elif len(xs) == 2:
            # Two distinct states give a direction but no error bar: trust it
            # only a quarter, and carry extra spread.
            b = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
            move = 0.25 * b * (x_target - xs[-1])
            median_logit = last_logit + max(-MAX_TREND_MOVE,
                                            min(MAX_TREND_MOVE, move))
            pred_var = 0.3 ** 2

        spread_sq = rw * rw + pred_var + (0.3 * shock) ** 2
        if flat:
            spread_sq += FLAT_SPREAD ** 2
        spread = min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(spread_sq)))
        qs = []
        for z in Z:
            q = round4(clamp_pct(inv_logit(median_logit + z * spread)))
            qs.append(q)
        # Keep the row non-decreasing after rounding.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
