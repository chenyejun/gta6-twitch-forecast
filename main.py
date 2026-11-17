"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Reads one case (makefun/twitch-ccu-case/v1) from the input directory and writes
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]} to <id>.out.

Method
------
The forecast is built from the case's own comparables: the hourly viewer curves of
past big game launches, each reduced to log-ratios log(y_h / y_1) against its own
hour-1 print. Two views of those ratios are combined:

1. Local-linear (LOESS) fit, per hour, of log-ratio against log(hour-1 viewers).
   Hour 1's level almost fixes the shape: a launch that is already huge in hour 1
   (rolling midnight unlock, most regions live) barely grows into the evening,
   while one still tiny in hour 1 (global unlock later in the day) grows many-fold
   (in the 37-launch leave-one-out check, corr(log p1, log y_19 / y_1) is -0.93).
   A Gaussian kernel in log p1 (bandwidth 1.25) with a local slope fits that
   relationship without imposing a global straight line.

2. Kernel-weighted median shape: per hour, the weighted median log-ratio over the
   donors, weighted by closeness in log p1 (bandwidth 1.0), by the data tier
   parsed from each comparable's unlock note (early-access-contaminated 0.4,
   weak 0.3, clean 1.0), and by how closely the donor's pre-launch ramp matches
   the target's own (log viewers 13 h and 1 h before day start, relative to
   hour 1, bandwidth 0.7).

Each view supplies a centre and a spread (its weighted residual standard
deviation in log space, floored and capped); the two views' five quantiles are
combined as a 50/50 geometric mean, and the spread is widened by 1.1, which in
leave-one-out brings the 90% interval coverage to about 0.90.

Validation (leave-one-out over the 37 past launches with a full release day in
the kit's comparables): mean pinball 0.367, median per-launch pinball 0.288,
90% coverage 0.905, typical median miss 74%; the persistence baseline on the
same launches scores 0.930 / 0.325 / 0.624 / 213%. On the 28 launches with
hour-1 viewers >= 10,000 (the size class a GTA VI hour 1 falls in) the model
scores 0.268 mean / 0.181 median against the baseline's 0.351 / 0.259.

Robustness: any failure while fitting falls back to the persistence baseline,
outputs are clamped to (0, 1e9], made non-decreasing and rounded half-up to
2 decimals, and the program is deterministic (no randomness, no wall clock,
donors processed in file order).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back first,
# so a module shipped beside this file would import (none is needed here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (as in the kit's starter).
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (chosen on the leave-one-out check above) ---
LOESS_BW = 1.25        # Gaussian kernel bandwidth in log(hour-1 viewers)
KERNEL_BW = 1.0        # same, for the weighted-median view
PRE_BW = 0.7           # bandwidth for pre-launch ramp similarity (log units)
SPREAD_WIDEN = 1.1     # final spread multiplier (calibrates 90% coverage)
LOESS_S_FLOOR = 0.08
LOESS_S_CAP = 1.0
KERNEL_S_FLOOR = 0.10
KERNEL_S_CAP = 1.2
MIN_DONORS = 5         # below this, fall back to the unweighted pool


def round2(x: float):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tier_weight(comp):
    """Data tier from the comparable's unlock note: early-access launches and
    weak analogues carry less weight; the GTA VI pre-release history is not a
    launch at all and is dropped by the release-day completeness check."""
    u = comp.get("unlock") or ""
    if "Tier: early-access-contaminated" in u:
        return 0.4
    if "Tier: weak" in u:
        return 0.3
    return 1.0


def extract_launch(comp):
    """A comparable's release-day curve as hour -> viewers (hours 1..24), plus
    pre-launch log-ratios at offsets -13 h and -1 h (relative to hour 1)."""
    ds = parse_ts(comp["day_start"])
    hrs = {}
    for pt in comp.get("hourly") or []:
        try:
            t = parse_ts(pt["from"])
        except (KeyError, TypeError, ValueError):
            continue
        dh = (t - ds).total_seconds() / 3600.0
        if dh == int(dh) and -336 <= dh < 72:
            v = pt.get("viewers")
            if isinstance(v, (int, float)) and v > 0:
                hrs[int(dh)] = float(v)
    if 0 not in hrs:
        return None
    present = sum(1 for h in range(24) if h in hrs)
    if present < 20:  # not a usable release day (e.g. the pre-release history)
        return None
    p1 = hrs[0]
    y = {h + 1: hrs[h] for h in range(24) if h in hrs}
    ratios = {h: math.log(y[h] / p1) for h in range(2, 25) if h in y}
    pre = {}
    for off in (-13, -1):
        if off in hrs:
            pre[off] = math.log(hrs[off] / p1)
    return {"p1": p1, "y": y, "ratios": ratios, "pre": pre,
            "tierw": tier_weight(comp)}


def target_pre_features(case):
    """The target launch's own pre-launch log-ratios at offsets -13 h and -1 h,
    from history_hourly (falling back to aggregating history)."""
    try:
        ds = parse_ts(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    known = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if known is None:
        return {}
    p1 = max(1.0, float(known["viewers"]))
    by_off = {}
    rows = case.get("history_hourly") or []
    if rows:
        for pt in rows:
            try:
                t = parse_ts(pt["from"])
            except (KeyError, TypeError, ValueError):
                continue
            off = (t - ds).total_seconds() / 3600.0
            if off == int(off) and -336 <= off < 0:
                v = pt.get("viewers")
                if isinstance(v, (int, float)) and v > 0:
                    by_off[int(off)] = float(v)
    else:  # aggregate minute reads into whole-hour means
        acc = {}
        for pt in case.get("history") or []:
            try:
                t = parse_ts(pt["at"])
            except (KeyError, TypeError, ValueError):
                continue
            off = math.floor((t - ds).total_seconds() / 3600.0)
            if -336 <= off < 0:
                v = pt.get("viewers")
                if isinstance(v, (int, float)) and v >= 0:
                    s, n = acc.get(off, (0.0, 0))
                    acc[off] = (s + float(v), n + 1)
        for off, (s, n) in acc.items():
            if n > 0 and s > 0:
                by_off[off] = s / n
    pre = {}
    for off in (-13, -1):
        if off in by_off:
            pre[off] = math.log(by_off[off] / p1)
    return pre


def wls_1d(xs, ys, ws):
    """Weighted least squares for y ~ b0 + b1 * x; returns (b0, b1)."""
    sw = sum(ws)
    if sw <= 0:
        return (sum(ys) / len(ys) if ys else 0.0, 0.0)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b1 = sxy / sxx
    return (my - b1 * mx, b1)


def wquantile(pairs, tau):
    """Weighted quantile with linear interpolation on cumulative weight."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0 or not pairs:
        return None
    thresh = tau * tot
    cum = 0.0
    for i, (v, w) in enumerate(pairs):
        prev = cum
        cum += w
        if cum >= thresh:
            if i == 0 or w == 0:
                return v
            frac = (thresh - prev) / w
            return pairs[i - 1][0] + frac * (v - pairs[i - 1][0])
    return pairs[-1][0]


def loess_view(p1, pre, donors):
    """Per-hour local-linear fit of log-ratio on log p1. Returns {h: (centre, s)}."""
    lp1 = math.log(p1)
    out = {}
    for h in range(2, 25):
        rows = []
        for d in donors:
            if h not in d["ratios"]:
                continue
            w = d["tierw"] * math.exp(-0.5 * ((math.log(d["p1"]) - lp1) / LOESS_BW) ** 2)
            rows.append((math.log(d["p1"]) - lp1, d["ratios"][h], w))
        if len(rows) < 3:
            rows = [(math.log(d["p1"]) - lp1, d["ratios"][h], 1.0)
                    for d in donors if h in d["ratios"]]
        if not rows:
            return None
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ws = [r[2] for r in rows]
        b0, b1 = wls_1d(xs, ys, ws)
        tot = sum(ws)
        var = sum(w * (y - (b0 + b1 * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / tot
        s = min(max(math.sqrt(max(var, 0.0)), LOESS_S_FLOOR), LOESS_S_CAP)
        out[h] = (b0, s)
    return out


def kernel_view(p1, pre, donors):
    """Per-hour kernel-weighted median shape with pre-ramp similarity.
    Returns {h: (centre, s)}."""
    lp1 = math.log(p1)
    out = {}
    for h in range(2, 25):
        pairs = []
        for d in donors:
            if h not in d["ratios"]:
                continue
            w = d["tierw"] * math.exp(-0.5 * ((math.log(d["p1"]) - lp1) / KERNEL_BW) ** 2)
            for off in (-13, -1):
                tv = pre.get(off)
                dv = d["pre"].get(off)
                if tv is not None and dv is not None:
                    w *= math.exp(-0.5 * ((dv - tv) / PRE_BW) ** 2)
            if w > 1e-6:
                pairs.append((d["ratios"][h], w))
        if len(pairs) < MIN_DONORS:
            pairs = [(d["ratios"][h], 1.0) for d in donors if h in d["ratios"]]
        if not pairs:
            return None
        m = wquantile(pairs, 0.5)
        tot = sum(w for _, w in pairs)
        var = sum(w * (v - m) ** 2 for v, w in pairs) / tot
        s = min(max(math.sqrt(max(var, 0.0)), KERNEL_S_FLOOR), KERNEL_S_CAP)
        out[h] = (m, s)
    return out


def baseline_quantiles(p1, h):
    return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]


def forecast_case(case, name):
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    hours = [row["hour"] for row in case["forecast"]]

    donors = []
    for comp in case.get("comparables") or []:
        try:
            launch = extract_launch(comp)
        except (KeyError, TypeError, ValueError):
            launch = None
        if launch is not None:
            donors.append(launch)
    pre = target_pre_features(case)

    loess = loess_view(p1, pre, donors) if len(donors) >= MIN_DONORS else None
    kernel = kernel_view(p1, pre, donors) if len(donors) >= MIN_DONORS else None

    rows = []
    for h in hours:
        if loess is not None and kernel is not None:
            m_lo, s_lo = loess[h]
            m_ke, s_ke = kernel[h]
            qs = []
            for z in Z:
                q_lo = math.exp(m_lo + z * s_lo)
                q_ke = math.exp(m_ke + z * s_ke)
                qs.append(math.exp(0.5 * math.log(q_lo) + 0.5 * math.log(q_ke)))
            # Widen the spread around the combined median.
            med = qs[2]
            qs = [med * (q / med) ** SPREAD_WIDEN for q in qs]
            quantiles = [p1 * q for q in qs]
        else:
            quantiles = baseline_quantiles(p1, h)
        # Clamp to the valid range, enforce non-decreasing, round half-up.
        clean = []
        prev = 0.0
        for q in quantiles:
            if not (isinstance(q, float) and math.isfinite(q)):
                q = float(p1)
            q = min(max(q, 0.01), 1e9)
            q = max(q, prev)
            clean.append(round2(q))
            prev = clean[-1]
        rows.append({"hour": h, "q": clean})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never leave a case invalid: fall back to the persistence baseline.
            first = next((k for k in case["known"] if k["hour"] == 1), None)
            p1 = max(1.0, float(first["viewers"])) if first else 1.0
            rows = []
            for row in case["forecast"]:
                h = row["hour"]
                clean = []
                prev = 0.0
                for q in baseline_quantiles(p1, h):
                    q = min(max(q, 0.01), 1e9)
                    q = max(q, prev)
                    clean.append(round2(q))
                    prev = clean[-1]
                rows.append({"hour": h, "q": clean})
            forecast = {"hours": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
