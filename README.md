[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/takhao666/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/takhao666/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy-rarity forecast

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at the case's target, as five quantiles (5/25/50/75/95%).

## How it works

`main.py` (Python 3.13, standard library only) reads every `*.in` file in
`$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (else `/output/`). Two regimes, picked from what the
case's history shows:

- **Flat history** (the practice cases). Every read lists the same players
  tracked and the same rates: the page is stale. By the target a refresh
  reveals the players who joined in the meantime, so a trophy's rate is
  diluted: `rate = r * (1 + (G - 1) * u) / G`, where `G` is the tracked-player
  growth since the first read and `u` is the share of the trophy's old earn
  rate newcomers reach, growing with how early/easy the trophy is (proxied by
  its own launch-night rate). The 100% trophy stays at 100%.
- **Moving history** (the final). Exophase has refreshed, so the reads trace
  real growth. Per trophy, a line is fitted to `logit(rate)` against
  `ln(1 + effective hours since release)` — weekend hours count 1.2x, the
  final's window holds a full weekend — with the slope shrunk towards the
  tier's median slope, anchored on the last read, floored at 0 for horizons
  over a day. Where the history shows the curve bending over, a concave
  quadratic's increment is used instead, so a saturating story trophy is not
  extrapolated to 100%. A tier cap stops runaway growth. The spread combines
  a random-walk term growing with the horizon, the fit's residual and a
  refresh-lag term.

## Validation

The kit ships no local grader and no practice answers (the server grades),
so the practice cases were scored against synthetic answers built from the
facts in `RULES.txt` (04:30Z read: 3,868 players, every rate but the 100%
one lower), calibrated so the house baseline scores its published ~1.51:

| case | scenario | house baseline | this entry |
|------|----------|----------------|------------|
| p00 | central dilution | 1.407 | **0.141** |
| p01 | central dilution | 1.550 | **0.145** |

Across eight further answer scenarios (stronger/weaker dilution, class-based
newcomer earn rates, random per-trophy rates, different player growth) the
entry scores 0.12–0.25 against the baseline's 0.98–1.78. On a synthetic
final-like case (mechanistic cohort simulation over 2.5 days of reads, target
4.5 days out) it scores 0.219 against the baseline's 0.460, with 90%
coverage 1.00. Outputs are deterministic and run in ~0.1 s per case.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
