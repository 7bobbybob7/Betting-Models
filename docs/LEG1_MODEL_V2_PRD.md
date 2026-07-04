# Leg 1 Model v2 — PRD: Can our own model beat the market?

**Status:** EXECUTED — Attacks 1, 4, 2 failed; kill criterion was briefly met, then the
new-data path was executed same-week (Statcast pitch-extras backfill: catcher ID, spray,
bat tracking — 1.84M pitches 2024-26) and **Attack 3 PASSED: first positive residual of
the campaign. TB clears every pre-registered gate in BOTH time directions.**
Leg 1 un-shelved, scoped to the new-data (v3) features. See §10 Results.
**Author:** modeling
**Date:** 2026-06 (in-season); results 2026-07
**Owner:** Leg 1 (own predictive model)

---

## 1. Background — why v1 failed, precisely

Leg 1 v1 trained a per-batter-game classifier (HRR/TB/RBI over a line) on 112 leak-safe
features (batter/pitcher arsenal, park, lineup, matchup). It does **not** beat the market.
We established this rigorously, not by vibes:

| Finding | Evidence |
|---|---|
| Model is a hair *worse* than the market at ranking | model AUC ~0.54–0.55 vs market de-vig AUC ~0.56–0.57 |
| Our features add **nothing** on top of the market line | market+model ≈ market alone (fit blend on 2025, test 2026) |
| Bottleneck is information, not method | distillation (train on Novig's price) hit the **same** 0.55 ceiling |
| The market *does* have lazy corners | K-suppression archetype mispriced ~1.4%, stable across 2025 **and** 2026 |
| Betting population is a coin-flip clump | 97% of HRR-1.5 props sit in market-implied 40–60%; std only 0.049 |

**Conclusion that drives v2:** a better model of *stable player quality* is worthless — the
market prices all slow public aggregates. To beat it we must attack surfaces the market
prices **lazily, late, or not at all**, and concentrate bets where multiple such signals
agree. The K-suppression finding is the existence proof that such corners exist.

## 2. Goal & non-goals

**Goal:** produce a model whose probability contains information **orthogonal to the market
line**, enough that a vig-clearing subset of bets exists — measured by the residual test (§6),
not by standalone AUC.

**Primary success metric:** `AUC(market + model) − AUC(market alone) > 0` on out-of-sample
data, ideally ≥ +0.005, concentrated in identifiable segments. Secondary: paper-trade ROI of
the flagged subset clears the venue's vig forward.

**Non-goals:**
- Not trying to raise standalone AUC for its own sake (v1's mistake).
- Not re-attempting slow/stable features (proven subsumed by the market).
- Not solving venue/execution here — that's the Leg 2 / venue work. This PRD is purely
  "does our number know something the line doesn't."

## 3. Guiding hypotheses (what the market is lazy about)

The prop market prices ~250 hitters × ~13 stat types × ~15 games *daily*. It cannot price
every game-specific context with the rigor it applies to 15 game totals. We target four
surfaces, each with a mechanical reason the book underweights it:

- **H1 — Fresh game context.** Umpire zone, bullpen fatigue, batter rest. These move
  K/BB/production and change daily; totals markets price umps, prop markets plausibly don't
  bother per-hitter.
- **H2 — Ladder incoherence.** Books post multiple lines per stat (0.5/1.5/2.5; Kalshi 1+…5+)
  and price each semi-independently → the implied count distribution is occasionally
  incoherent. Exploiting this needs *zero* informational edge, only that the book's ladder is
  sloppier than a distribution family.
- **H3 — Newest data lag.** Statcast bat-tracking (swing speed/length/attack angle,
  squared-up) exists only since 2024 — the data the market has had least time to absorb.
- **H4 — Population mismatch.** We trained on all starters (easy scrub-vs-star separation)
  but bet only the 40–60% clump. Training on the betting population may sharpen discrimination
  where it actually matters.

## 4. Scope — four "attacks", sequenced

Ranked by effort-to-promise. Each is independently shippable and independently falsifiable.

### Attack 1 — Fresh game-context features  *(build first)*
New module `models/mlb/game_context_features.py`. All strict `< as_of_date`, `closed='left'`.

| Feature | Source | Mechanism | Data status |
|---|---|---|---|
| `ctx_ump_k_rate_365d`, `ctx_ump_bb_rate_365d` | `mlb_game_info.umpire_hp` + box outcomes | big-zone umps inflate K, suppress production | ✅ 100% coverage 2022+ (~90 umps; 365d ≈ 60 games ≈ 4,800 PA per ump) |
| `ctx_ump_runs_pg_365d` | ump + game runs vs league | some umps run high/low scoring | ✅ |
| `ctx_opp_bullpen_relievers_1d` | `mlb_pitching_game` reliever appearances, prior day | gassed pen → more late production | ✅ derivable (177K reliever-games) |
| `ctx_opp_bullpen_ip_2d`, `_ip_3d` | reliever IP prior 2/3 days | cumulative fatigue | ✅ |
| `ctx_batter_rest_days`, `ctx_batter_games_7d` | game dates | fatigue depresses output | ✅ |
| `ctx_catcher_framing_*` | `mlb_pitches` edge takes | framing shifts K/BB | ⚠️ **blocked** — no catcher ID in `mlb_pitches`; needs backfill. Deferred to a follow-up. |

Deliverable: ~8 features, merged onto the cached parquets by `(player_id, game_id)` — no full
dataset rebuild needed.

**Production note:** the HP umpire's *identity* is announced day-of-game. Not a backtest leak
(the feature value is his rolling historical tendency, strictly `< as_of_date`; his identity
is game metadata known before first pitch) — but it means live predictions using ump features
are day-of only.

### Attack 4 — Train on the betting population  *(cheap bolt-on)*
Filter/reweight training rows toward the near-coin-flip clump (market-implied 0.40–0.60, or
where a real prop line existed 2024+). Tests whether v1's edge was diluted by learning easy
separations it never bets. No new data.

### Attack 2 — Distributional ladder-coherence  *(most original)*
New module `models/mlb/ladder_coherence.py`. Per player-game, fit a count distribution
(Negative Binomial / compound Poisson) **anchored to the market's own 1.5 price** (so we do
not claim to out-know the mean), then flag rungs (0.5/2.5, or Kalshi 1+…5+) whose implied
prob deviates from the coherent curve beyond a threshold. Uses the Kalshi full ladder
(order books now readable) and Underdog alts. Edge source: the book's internal inconsistency,
not our forecasting.

### Attack 3 — Bat-tracking swing data  *(big investment, gated on 1/2 showing life)*
New scraper for Statcast bat-tracking (2024+): swing speed, swing length, attack angle,
squared-up rate, per batter. New batter swing-profile features. Raises the ceiling for both
Leg 1 and Leg 3 (distillation) if the fresh-data lag hypothesis holds.

## 5. Data plan

- **Available now (no new pulls):** umpire, bullpen, rest (Attacks 1, 4).
- **New pulls required:** catcher IDs for framing (Attack 1 follow-up); bat-tracking (Attack 3).
- **Leak discipline:** every feature computed strictly from `game_date < as_of_date`;
  umpire/bullpen tendencies use rolling 180d `closed='left'`; reuse the cached
  train (2019–2024) / backtest (2025–2026) parquet split.
- **Label:** unchanged — HRR>1.5 (primary), TB>1.5, RBI>0.5, from box scores.

## 6. The decisive test (definition of success)

This is the gate. It is the exact test that killed v1's slow features:

1. Build feature set = **existing 112 + new fresh-context**. Retrain XGB (locked winner
   hyperparameters, no re-search) on 2019–2024 outcomes.
2. Score → `p_model_v2`. Get market de-vig prob → `p_mkt`. Score the **old** v1 model →
   `p_model_v1` (control).
3. Fit three logistic blends on a time-split (fit 2025, test 2026):
   - A: `outcome ~ p_mkt`                     (market alone)
   - B: `outcome ~ p_mkt + p_model_v2`        (the candidate)
   - C: `outcome ~ p_mkt + p_model_v1`        (control — known to add ≈ 0)
4. **Pass iff `AUC_B − AUC_A > 0`** (target ≥ +0.005) **and** `AUC_B > AUC_C` (lift
   attributable to the new features, not refit noise). Also report the `p_model_v2`
   blend coefficient.
5. Segment the lift by the **pre-registered segments below** — a lift that concentrates where
   the mechanism predicts, consistently in 2025 and 2026, is trustworthy; an aggregate-only
   blip is not.

**Pre-registered segments (defined BEFORE looking at results — do not add segments later):**

| Segment | Definition | Predicted direction |
|---|---|---|
| S1 big-zone ump × high-K pitcher | `ctx_ump_k_rate_365d` top tercile AND `pit_k_rate_szn` top tercile | market overprices overs → unders outperform |
| S2 gassed opposing pen | `ctx_opp_bullpen_ip_2d` ≥ 7 innings | market underprices overs → overs outperform |
| S3 no-rest batter | `ctx_batter_games_7d` ≥ 7 (no day off in a week) | market overprices overs → unders outperform |

If Attack 1 fails this test, fresh context is *also* subsumed by the line → strong signal the
market is efficient even on game context, and we stop pouring effort into Leg 1.

## 7. Risks & how we avoid fooling ourselves

- **Data-snooping the segments:** pre-register segment definitions in this doc before looking;
  require the same sign/direction in both test seasons.
- **Leakage:** fresh-context features are the highest leak risk (umpire/bullpen "as of" the
  game). Enforce `< as_of_date` and unit-test on a known case.
- **Small real edge < vig:** even a passing residual test may be sub-vig (like K-suppression's
  1.4%). Success at the *model* layer ≠ profitable; it becomes a **prior that stacks** with
  Leg 2 line-shopping and the K-suppression archetype. Stacking independent sub-vig signals
  is the path to a vig-clearing bet.
- **Effort sink:** Attacks 3 (bat-tracking) gated on Attack 1/2 showing life. Fail fast.

## 8. Sequence & exit criteria

1. **Attack 1** (umpire + bullpen + rest) → residual test. *Exit if no lift after adding
   framing (follow-up) — declare game context priced.*
2. **Attack 4** (population reweight) → cheap, run alongside.
3. **Attack 2** (ladder coherence) → independent of 1; uses Kalshi ladder.
4. **Attack 3** (bat-tracking) → only if 1 or 2 clears the residual test.

**Overall kill criterion:** if Attacks 1, 2, and 4 all fail the residual test across both
seasons, Leg 1 is not revivable with available data; concentrate on Leg 2 (line-shopping vs
soft singles books) and shelve own-model work pending new data sources.

## 10. Results (2026-07)

**Attack 1 — fresh game context: FAIL.**
Residual test (HRR, fit 2025 → test 2026): A market-alone **0.5587**, B market+v2 0.5538,
C market+control 0.5568. B−A = −0.005, B−C = −0.003 — the fresh-context model *subtracts*
from the line. Pre-registered segments all failed two-season consistency:
S1 flipped sign (2025 +0.021 / 2026 −0.046 z=−2.9 — would have looked like a discovery
without pre-registration), S2 null both years, S3 significant in the *wrong* direction in
2026 (+0.028, z=+2.1). Umpire/bullpen/rest, as measured, are priced.

**Attack 4 — betting-population training: FAIL (weakly testable).**
Clump-trained arms D 0.5538 / E 0.5549 vs A 0.5587. Caveat: the clump filter only removed
23% of rows — model predictions are naturally compressed into the clump, so the hypothesis
had little room to act.

**Attack 2 — ladder coherence → revised to RUNG TRANSLATION: FAIL by pre-registered
consistency, but the most alive of the three.**
Premise revision: Underdog posts NO within-player ladders (0.5/1.5/2.5 are different
players), so within-book incoherence doesn't exist; the attack became "translate the sharp
1.5 anchor to fair 0.5/2.5 prices via empirical curves, test UD's tail rungs."
Curves: clean, monotone, anchor well-calibrated (13.3K scored props).
Backtest: clean out-of-time split (curve Apr–May → test Jun) **−9 to −12% ROI** at all
thresholds (n=193). Full-sample day-parity cross-fit (n=1,024): +6.0%±7.0% at ev>4%
(n=499) — but decomposes to Apr–May +11% / Jun −12%: a period sign-flip, failing the same
consistency bar that killed S1. Not bettable on this evidence. Worth passive forward
monitoring; the rung-translation machinery also expands Leg 2's matchable universe.

**Kill criterion (§8) was met on the original three attacks** — then the revival path was
executed immediately rather than deferred:

**Data expansion (2026-07-03/04):** all "missing" datasets turned out to be columns in the
Statcast feed we never stored. Backfilled `mlb_pitch_extras` 2024→today (1.84M pitches):
catcher ID 100%, bat_speed/attack_angle 45-46% (= swing rate ✓), hc_x/hc_y 17-18%
(= BIP rate ✓). Daily cron tops it up (rolling 4d). Ops lessons hardened into the script:
per-chunk subprocess timeouts (in-process SIGALRM was defeated by pybaseball's threads)
and execute_values batching (executemany = 1 round-trip/row = 23 min per 16K rows).

**Attack 3 — new-data features (v3): PASS. First positive residual of the campaign.**
7 features (models/mlb/advanced_profile_features.py): true pull%/oppo% (spray coords),
bat speed / swing length / attack angle / fast-swing rate (120d), opposing-catcher framing
(edge-take called-strike rate, 200d). All closed='left'. Residual test (fit 2025→test 2026
and reverse):

| target | B−A (25→26) | B−C (25→26) | B−A (26→25) | B−C (26→25) | verdict |
|---|---|---|---|---|---|
| TB  | **+0.0052** | +0.0028 | **+0.0121** | +0.0042 | **PASS both directions, meets +0.005** |
| HRR | +0.0013 | +0.0032 | — | — | weak pass |
| RBI | −0.0006 | +0.0025 | — | — | marginal fail |

B−C (new features vs identical control) positive in ALL five tests (~+0.003 each) — the
features add real information on every target. B−A ordering matches mechanism: strongest
on TB (power/contact target = what bat speed & pull measure), weakest on RBI (most
lineup-context-dependent). First candidate of the campaign to clear the two-direction
consistency bar that killed S1 and rung translation.

Real-pull segment (user's archetype, RHB pull-hitter vs sinker-heavy LHP): 2026 unders
beat market by 9.4 pts (n=78, z=−1.7) — direction matches mechanism, sample still small;
track forward, don't bet yet.

**Disposition: Leg 1 un-shelved, scoped to v3.** Next: harden the v3 TB model (calibrate,
save bundle), track its disagreements with the market in the forward paper-trade next to
Leg 2, and iterate the feature set (pull-air rate, platoon-split pull, framing×K
interactions). Sizing reality: +0.005-0.012 blend AUC is a real informational edge but
thin vs vig — its near-term value is bet FILTERING on Leg 2 (with the thrice-confirmed
K-suppression family), not standalone betting.

## 9. Open questions

- Catcher-ID backfill: source (MLB API play-by-play has catcher per pitch) and cost.
- Bat-tracking history depth (2024+ only) — enough for training?
- For Attack 2, is Underdog's alt-line ladder rich enough, or is Kalshi's 1+…5+ the only
  usable ladder? (Kalshi is sharp, so incoherence may be smaller there.)
