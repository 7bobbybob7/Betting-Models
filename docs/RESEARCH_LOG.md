# Research Log

Every experiment, its verdict, and where it lives. Newest campaigns last.
**Methodology (applies throughout):** features/ideas are tested in pre-declared batches
through a *gate*: candidate vs control (current production stack) vs market-alone, on
out-of-time splits in **both directions**. ACCEPT requires beating both, both directions.
Segments/trades additionally require **cross-period sign consistency**. Accepted features
are promoted to `feature_sets.py` (MLB) or bundles; gate scripts stay frozen in
`research/`. Backtests suggest; **forward trackers decide** (all cron-fed, settle daily).

---

## MLB Leg 1 — hitter model campaign (2026-07-03 → 07-05)

| Experiment | Hypothesis | Verdict | Notes |
|---|---|---|---|
| Attack 1: game context (ump zone, bullpen fatigue, batter rest) | markets price context lazily | ❌ REJECT | blend *below* market; all 3 pre-registered segments flipped across seasons |
| Attack 4: clump-trained (betting-population reweight) | training on bettable rows sharpens | ❌ REJECT | filter only removed 23% of rows — hypothesis barely testable |
| Attack 2: rung translation (anchor 1.5 → price 0.5/2.5) | tail rungs set lazily | ❌ REJECT | out-of-time −12%; pooled +6% dissolved into period sign-flip |
| **Attack 3: advanced profile (spray/pull, bat-tracking, framing)** | newest data least absorbed | ✅ **ACCEPT** | first positive residual; TB both directions; B−C +0.003 on all targets |
| **Batch 1: pull-air, smash factor, platoon pull, speed-vs-95** | mechanism refinements | ✅ **ACCEPT (v4)** | smash factor first adv feature in top-20 importance |
| Batch 2: swing-change deltas (30d vs 120d) | books anchor on stale profiles | ❌ REJECT | level features already carry changes (rolling windows update daily) |
| **Batch 3: luck gap (actual − xBA/xSLG on contact)** | books anchor on outcomes not deserved | ✅ **ACCEPT (v5)** | full 2019+ coverage; TB both directions |
| Embeddings v1: explicit batter×pitcher crosses | matchup interactions | ❌ REJECT | L1 kept 58/64 crosses (no sparsity) — trees already had it |
| Embeddings v2: latent MF on PA outcomes | pair chemistry beyond profiles | ❌ REJECT | factors shrank to std 0.0003 — **BvP chemistry = noise** (2-6 PA/pair) |
| **v6: same features at FULL 2019-2026 coverage** | coverage is the multiplier | ✅ **ACCEPT (both targets)** | pull coverage 13%→82%; 5 adv features into top-20; edge ~2× |
| Batch 4: swing-path/arm-angle levels | unused stored columns | ❌ REJECT | redundant with attack angle + pull profile |
| HR market gate (Novig anchor) | HR anchored on trailing counts | ✅ **PASS 4/4** | 27K props; long odds amplify thin edges |
| Pitcher K gate (archived Poisson + modern feats) | K-suppression mechanism | ❌ REJECT | **flagship K market is sharp** (mkt 0.58 vs model 0.54); edge lives in derivative markets |
| Pitcher OUTS v1→v2→v3 | outs = manager DECISION (leash) | 🚧 FROZEN | decision family +0.032 (biggest single-batch jump); line-conditional architecture invented here; ends at blend boundary |
| Line-conditional for TB | transfer outs architecture | ❌ REJECT | easy 0.5-rungs diluted the 1.5 boundary |

**Betting-economics audits (the honesty stack):**
- Standalone sims: fail two-year bar on TB/HRR/HR — traced to **side floods** (95% U 2025 / 86% O 2026) from blend intercept transfer.
- Noise-placebo: model-selected unders +6% vs noise-selected +0.8% (same side/year) — **selection skill is real**.
- Blanket unders: lose everywhere — **shade < vig** at every book.
- Calibration: **universal over-shade 0.5-2.5pts, 13/13 cells** (even Novig), shrinking yearly (2.3→1.5→0.6).
- **Adaptive blend (90d walk-forward)**: kills side floods; v5 standalone ≈ 0; **v6 standalone positive both years at ev>2%** (+3.7/+7.3% at thresholds) — pre-registered live candidate.
- **Filter/veto role**: agree-vs-disagree split validated 3 model versions, both years (disagree bets −7 to −44%).
- v7 production retrain (thru 2026-04): +63K bat-tracking rows; first HR bundle; live July+ honest.

## MLB Leg 2 — venue sweeps (2026-07-05 → 07-11)

| Test | Verdict |
|---|---|
| Underdog vs Novig (paper-trade) | live; mid-price fix (last→mid) was the critical repair |
| DK/Caesars/Fanatics/BetMGM sweep (all markets) | **DK RBI +8.5%, RUNS +11.9%, OUTS +14.8-18.5% — month-consistent survivors** → 3-market tracker |
| Fliff | ❌ vig wall (11.5%); EV threshold test FLAT at all levels (DK positive control rose monotonically) — **no threshold rescues a vig wall** |
| Kalshi (earlier) | sharp ≈ Novig; no edge |
| Cross-book RBI confirmation (Apr-Jun) | June multi-book signal dissolved; **only DK consistent 3/3 months** — book-specific softness, not market-wide |

## WNBA campaign (2026-07-12 → 07-18)

| Test | Verdict |
|---|---|
| 14-book backfill (143K props, 2025-05→now) | complete; BettingPros WNBA archive starts at 2025 season opener |
| Anchor referee (Brier vs outcomes, head-to-head) | Novig best-but-barely; **disagreements = coin flip (50.9%)** → **no sharp-vs-soft edge exists in WNBA** |
| Structural: universal ~3pt over-shade vs vig | **Novig blanket unders** (3PM/AST/PTS) +2.6%, **+3.4% best-price-routed**; period-consistent (weak 2025H2, strong 2026H1) → tracked |
| Venue verdicts | retail 7% vig ≈ cancels shade; Sleeper/PrizePicks 12-15% = vig walls |
| Model v1 (box-score rolling) | ❌ all four markets lose to a 0.54-0.58 market |
| **v2: minutes/rotation decision family** | ✅ **ACCEPT (points)** — beats mkt both directions; two bug fixes en route (rebounds label: orb never populated, drb=totals; **games-table UTC date shift**: dual-date matching 51%→97%) |
| B3: shot-profiles + ESPN fingerprint (player-side) | ❌ already priced / proxied by box rollups |
| B4: opponent-lineup defensive profile (matchup) | ❌ but first-ever consistent rebounds lift (+both directions, sub-market) |
| Combo markets (PRA/PR/PA/RA) | ❌ noisier lines but our sums even noisier |
| B5: PBP tendencies (foul-trouble, leash, closer role, clean rates) | ❌ biggest rebounds lift (0.543→0.555, −0.002 from mkt) but flips |
| B4+B5 stacked | ❌ redundant (both proxy size×role) |
| **B6: realized player-vs-TEAM H2H (EB-shrunk residuals)** | ✅ **ACCEPT (points)** — 13-team density defeats MLB's sparsity; balanced +0.005/+0.006 over mkt → **wnba-v3** bundle, ranks the unders |
| Data banked | 95K shot charts; 556 fingerprint player-seasons (ESPN S3 unlock — per-SEASON files, unsigned); 620K PBP events (shufinskiy/nba_data) |

## Infrastructure incidents & fixes
- **executemany ≈ hang** (23 min/16K rows) → execute_values everywhere; subprocess-level timeouts (SIGALRM defeated by lib threads).
- **BettingPros never cronned** (data silently frozen at last manual pull) → daily cron + `--force`.
- **ON CONFLICT DO NOTHING froze settlements** → settlement-fields-only upsert; first-seen odds preserved (CLV entry), `closing_*` updated by later pulls; `scheduled_start` on all snapshot tables + execution-time stamps = delay-proof CLV.
- **Production bundles gitignored → CI trackers failed 19 straight days** (July 9-27, works-on-my-machine) → track production pkls; v3_signals gap repaired retrospectively (1,233 signals).
- **BettingPros began requiring their public frontend `x-api-key` (~Aug 7) → 403s on every pull, but the scraper swallowed them as "no props" and exited 0 → 6 silent dark days** (Aug 7-12, both sports, all books; only surfaced because the DK tracker crashed on the empty window). Fix: key added to HEADERS; 0-props-fetched + HTTP failures now exits 1 (loud red cron, never silent green); tracker got an empty-window guard; gap backfilled with `--force` and trackers re-run.
- **Same week, independently: ESPN started throttling GitHub runner IPs → WNBA games.py's full-season rescan (184 scoreboard calls) hung → job cancelled at the 25-min timeout 6 days straight (Aug 7-12), killing the downstream BettingPros + tracker steps too** — cancelled ≠ failed, so no red X pattern-matched. Game logs froze at Aug 4 → every points-unders row since scored without p_model. Fix: `--days` incremental mode (trailing-window scoreboard scan, `--days 7` in cron); logs backfilled locally, ranks repaired back through July.
- models/mlb reorg: `feature_sets.py` single source of truth; production never imports research.

## Circadian / travel investigation (2026-07-28)
- NFL west-coast night-game team edge: **does not replicate in MLB** (east-at-west night is *positive* for visitors — anti-circadian; long-trip confound).
- **Eastward-travel batter underperformance: REAL** — west batters on east trips run ~1.4pts under implied (excess of control), consistent 2025+2026, day AND night → mechanism is eastward jet-lag (phase advance), not game-time.
- As model features: ❌ REJECT (absorbed by rest/form features; 1.4pts × 7.7% of props ≈ invisible to blend AUC).
- Disposition: **selection heuristic** — skip overs / prefer unders on west batters early in eastern trips.
- Side product: `games.game_time` backfilled 2019-2026 (100%) from MLB Stats API.

## Standing production state (as of 2026-07-28)
- **Models:** MLB v7 TB + HR bundles (thru 2026-04); WNBA wnba-v3 points (minutes + H2H).
- **Tracked trades:** DK RBI/RUNS/OUTS singles; WNBA Novig unders (3PM/AST/PTS) with model ranking; UD paper-trade + veto.
- **Live candidates awaiting forward verdicts:** MLB v6/v7 walk-forward standalone (ev>4%); TB-unders-model-ranked; whiff×whiff TB pocket; DK singles; WNBA unders ± model rank.
- **Killed conclusively:** BvP chemistry, sharp-vs-soft in WNBA, Fliff/Sleeper/PrizePicks as venues, pitcher-K modeling, NFL-style circadian in MLB.
- **Open leads:** WNBA live lineup/injury feed; MLB catcher-framing follow-ups; monthly retrains as bat-tracking accrues.
