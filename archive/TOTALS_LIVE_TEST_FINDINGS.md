# Totals Live Test Findings (Apr 11 – Jun 9, 2026)

After ~60 days of live deployment, all 3 MLB totals models tested in
production were unprofitable. Sample size of 1,448 flagged bets across
3 strategies — large enough that results are statistically definitive,
not variance.

## Headline numbers

| Model | Bets | Win% | ROI | Wagered | P&L |
|-------|------|------|-----|---------|-----|
| Regression (≥1% info edge vs devig median) | 636 | 44.3% | **-11.5%** | $64,000 | -$7,349 |
| Classifier (≥3% info edge vs devig median) | 383 | 47.0% | **-6.3%** | $38,400 | -$2,430 |
| Best-Edge Classifier (≥1% edge vs best-odds implied) | 423 | 45.4% | **-9.6%** | $42,400 | -$4,060 |

Backtest predicted +0.3%, +1.7%, +2.6% respectively. Live performance was
12–14 percentage points worse than backtest across the board.

## The structural finding: OVER bias

Decomposing bets by direction revealed the core failure mode:

| Model | OVER bets | OVER Win% | OVER ROI | UNDER bets | UNDER Win% | UNDER ROI |
|-------|-----------|-----------|----------|------------|------------|-----------|
| Regression | 507 (80%) | 40.4% | -19.9% | 129 (20%) | 59.7% | **+21.3%** |
| Classifier | 247 (64%) | 38.1% | -24.6% | 136 (36%) | 63.2% | **+26.7%** |
| Best-Edge | 274 (65%) | 36.5% | -28.0% | 149 (35%) | 61.7% | **+24.2%** |

UNDER bets crushed (+21 to +27% ROI on 100+ samples each — statistically real).
OVER bets were catastrophic (-20 to -28% ROI on 250+ samples each — also real).

## Why this happened

The model's raw predictions were well-calibrated in aggregate:
- Avg predicted total: 8.99
- Avg actual total: 9.04
- Aggregate bias: only -0.05 runs

But the 2026 MLB run environment is significantly UNDER-favored:
- 50.1% of games went over 8.5 (vs market lines often set at 8.5)
- Only 39.8% of games went over 9.0 (vs market lines often set at 9.0)

The market line is consistently 0.3–0.5 runs higher than implied by actual
outcomes. Our model's `info_edge = model_p - devig_p` framework systematically
picks OVER because the model's P(over) ≈ market's P(over), and we end up
betting the same side as the market consensus — which is wrong in 2026.

## What this confirms from the 2024–2025 backtest

The 2024–2025 backtest already showed this trend was developing:
- 2024 regular-season ROI at ≥1% threshold: -2.6% (down from positive in 2019–2023)
- 2025 regular-season ROI at ≥3% threshold: -8.9%
- The market efficiency on totals tightened year over year

The 2026 live test just continued the trend, more severely.

## What we tried that didn't help

During the live testing period the model included:
- L1-regularized Logistic Regression with 22 candidate features
- Wind direction, umpire zone, bullpen availability (newly added features)
- Multiple training-window experiments (expanding, 3-year, 2-year rolling)
- De-vig framework with median consensus pricing
- Line shopping across 6 sportsbooks for best execution
- Postseason exclusion filter

None of these prevented the OVER-bias failure mode.

## Sample-size sanity

These are not noise:
- p-value for "UNDER bets hit above 52.4% breakeven" at 60%+ win rate over
  400+ samples is essentially zero
- p-value for "OVER bets hit below 50%" at 38–40% win rate over 1,000+ samples
  is essentially zero
- Only 6 of 1,448 bets are unresolved

## Lessons

1. **Backtest profitability does not guarantee live profitability** even with
   rigorous methodology (expanding-window CV, multiple seasons, proper de-vig).
   Market dynamics shift.

2. **An aggregate-calibrated model can still have a systematic bet-selection
   bias.** The predictions averaged 8.99 vs actuals 9.04 — almost perfectly
   calibrated. The damage came from the *selection* logic, not the predictions.

3. **The "model echoes market consensus" failure mode is real and measurable.**
   When our model picks the same side as the public, we lose. The edge calculation
   assumes we have private information; we don't.

4. **Decomposing performance by bet direction surfaces failure modes that
   aggregate metrics hide.** Aggregate ROI was bad. Direction breakdown showed
   one side was great and the other was destroying us — a much more actionable
   insight.

5. **Live test data is the only valid validation.** 60 days of live bets told
   us things 9 years of backtest data couldn't.

## Decision

Totals live test retired. Three models disabled. Focus shifts to MLB player
hitter props (HRR, Total Bases, Hits) via Underdog Fantasy.

## What carries forward to hitter props

- The 2-step evaluation framework (info edge → +EV gate) remains useful
- The decomposed CLV concept (Model CLV vs Execution CLV) remains valuable
- The expanding-window CV methodology remains rigorous
- The lesson that aggregate calibration doesn't imply bet-selection profitability
  will inform how we evaluate hitter prop models — track direction-level metrics
  from day one
