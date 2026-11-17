[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/chenyejun/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/chenyejun/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast — entry

<sub>Entered on [make.fun](https://make.fun/?ref=kr2w23qf&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

Predicts the hourly Twitch viewers of Grand Theft Auto VI on its release day
(2026-11-19 UTC), hours 2–24, as five quantiles per hour, from the first
hour's print and the case's comparables. Python 3.13, standard library only,
deterministic, no network.

## How it works

Each comparable (a past big launch's hourly curve) is reduced to log-ratios
`log(y_h / y_1)` against its own hour-1 print. The forecast combines two
views of those ratios, both computed at run time from the case's own
comparables (nothing is hard-coded to the practice set, so the final case's
42 comparables, including the four practice launches, are used as-is):

1. **Local-linear (LOESS) fit, per hour, of log-ratio on log hour-1 viewers.**
   Hour 1's level almost fixes the shape: a launch already huge in hour 1
   (rolling midnight unlock, most regions live) barely grows into the
   evening, while one still tiny in hour 1 (global unlock later in the day)
   grows many-fold. On the 37 launches with a full release day,
   corr(log p1, log y_19 / y_1) = -0.93. A Gaussian kernel in log p1
   (bandwidth 1.25) with a local slope fits that relationship without
   imposing one global straight line.

2. **Kernel-weighted median shape.** Per hour, the weighted median log-ratio
   over the donors, weighted by closeness in log p1 (bandwidth 1.0), by the
   data tier parsed from each comparable's unlock note
   (early-access-contaminated 0.4, weak 0.3, clean 1.0), and by how closely
   the donor's pre-launch ramp matches the target's own (log viewers 13 h
   and 1 h before day start, relative to hour 1, bandwidth 0.7 — those two
   hours are the New Zealand and central-Europe unlock hours).

Each view supplies a centre and a spread (its weighted residual standard
deviation in log space, floored and capped). The two views' five quantiles
are combined as a 50/50 geometric mean, and the spread is widened by 1.1,
which calibrates the 90% interval coverage to about 0.90.

## Why these parameters

Every number above was chosen on a leave-one-out check over the 37 past
launches in the kit's comparables that have a full release day (each launch
in turn forecast from the other 36). What was tried and dropped:

- A pooled median shape with no conditioning on hour-1 level: mean pinball
  0.78, far worse — the evening rise it predicts is far too big for large
  launches and far too small for small ones.
- A global linear regression of log-ratio on log p1: better than the pooled
  shape on large launches (0.269) but extrapolates badly on small ones
  (0.442 overall).
- The pre-launch diurnal amplitude (evening minus night viewers before
  release) as a predictor: correlation with the release-day shape is only
  ~0.09, so it is not used.
- Pre-ramp similarity inside the LOESS fit: it hurt there, so it is used
  only in the median view.

Final leave-one-out numbers (37 launches): mean pinball **0.367**, median
per-launch pinball **0.288**, 90% coverage **0.905**, typical median miss
**74%**. The persistence baseline on the same launches: 0.930 / 0.325 /
0.624 / 213%. On the 28 launches with hour-1 viewers >= 10,000 — the size
class a GTA VI hour 1 falls in — the model scores 0.268 mean / 0.181 median
against the baseline's 0.351 / 0.259.

## Files

- `main.py` — the whole entry. Reads every `*.in` in `$MAKEFUN_INPUT_DIR`
  (default `/input`), writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default
  `/output`). Any failure while fitting falls back to the persistence
  baseline for that case, so a case is never left invalid. Outputs are
  clamped to (0, 1e9], made non-decreasing and rounded half-up to 2 decimals.
- `LICENSE` — MIT, copied from the kit's starter.

## Local check

```
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

Runs in ~0.5 s for all four practice cases; two runs are byte-identical.
All four outputs pass the format check (23 rows, hours 2–24, five finite
non-decreasing quantiles each, file well under 65,536 bytes).
