# GTA VI Twitch CCU Forecast

[![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arwin0204/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arwin0204/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the [make.fun bounty gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecast hour-by-hour Twitch concurrent viewers for GTA VI's release day (2026-11-19 UTC) from the first hour's print, as five quantiles per hour (hours 2–24).

## Strategy

Per hour h = 2..24, an OLS regression in log space, fit at runtime from the case's own comparables:

```
log(viewers_h) = a_h + b_h * log(viewers_1)
```

- **Median** = the fitted line at the case's own hour-1 print.
- **Spread** = max(stdev of residuals, 1.4826 × MAD of residuals) × the prediction-interval leverage factor `sqrt(1 + 1/n + (x0 − x̄)²/Sxx)`, so the band widens honestly when the hour-1 print lies outside the training range (likely in the final: GTA VI's hour 1 may exceed every training hour 1).
- **Quantiles** from a normal distribution at the case's levels.

**Training filter (the key idea):** only launches whose release day had already started by hour 1 train the regression — those whose day peak is at most 20× the hour-1 print. A launch with a peak above 20× hour 1 had not started at hour 1 (a single global unlock later in the day, e.g. a PC game at 10:00/17:00 UTC, whose hour-1 print is only the pre-release baseline); including it flattens the slope and, on a large hour-1 print, predicts an absurd collapse.

**Fallbacks:** if fewer than 5 trained launches report an hour, use the median log-ratio to hour 1 over all comparables; if that is empty too, persist hour 1 with the house baseline's widening spread.

Why not persistence (the house baseline)? Hour 1 only holds the regions that already unlocked (New Zealand / Asia-Pacific); viewership dips as the world sleeps, then rises as Europe and the Americas unlock and enter prime time, peaking around hours 19–22 at ~1.5–2.5× hour 1 for a typical launch — and the rise is steeper when hour 1 is small (the OLS slope falls from ~1.05 in the early hours to ~0.65 in the afternoon). The regression captures both facts; the baseline does not.

## Verified practice results

Scored locally against the SullyGnome hourly averages (the kit's source), pinball loss on the log scale (lower is better):

| case | game | this model | house baseline |
|---|---|---|---|
| p00 | Marvel's Wolverine | 0.3037 | 0.7437 |
| p01 | Red Dead Redemption 2 | 0.2120 | 0.2868 |
| p02 | God of War Ragnarök | 0.1549 | 0.2146 |
| p03 | Marvel's Spider-Man 2 | 0.1915 | 0.3033 |
| **mean** | | **0.2155** | **0.3871** |

Beats the house baseline on **4/4 cases (−44.3% mean loss)**. Typical miss 70.3% / 45.8% / 26.3% / 32.2%; 90% coverage 0.87 / 0.96 / 1.00 / 0.91. Deterministic (two runs byte-identical); ~0.08 s per case (limit 8 s).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in the input directory and writes one `<id>.out` per case.
