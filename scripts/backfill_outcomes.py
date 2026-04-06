"""
scripts/backfill_outcomes.py - Backfill prediction outcomes, edges, and P&L.

Runs after games complete to:
1. Fill in outcomes (win/loss) for predictions where game is now final
2. Fill in edge for predictions that had no odds at prediction time
3. Compute P&L for flagged bets using flat $100 sizing

Should be run daily (or as part of the pipeline) to keep predictions table current.

Usage:
    python scripts/backfill_outcomes.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from db.db import query, execute


def american_to_decimal(american):
    if american is None or pd.isna(american):
        return 1.909
    if american >= 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def backfill_outcomes():
    """Fill in outcomes for completed games."""
    print("\n  Backfilling outcomes...")

    # Find predictions without outcomes where game is final
    pending = query("""
        SELECT p.prediction_id, p.game_id, p.model_name, p.market,
               p.predicted_prob, p.predicted_value, p.edge,
               p.bet_placed,
               g.home_score, g.away_score, g.status
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.outcome IS NULL AND g.status = 'final'
    """)

    if len(pending) == 0:
        print("    No predictions to backfill")
        return 0

    updated = 0
    for _, row in pending.iterrows():
        pid = int(row["prediction_id"])
        market = row["market"]
        home_score = row["home_score"]
        away_score = row["away_score"]

        outcome = None

        if market == "moneyline":
            if row["predicted_prob"] is not None:
                predicted_home = float(row["predicted_prob"]) > 0.5
                home_won = home_score > away_score
                outcome = "win" if predicted_home == home_won else "loss"

        elif market == "total":
            if row["predicted_value"] is not None and row["edge"] is not None:
                edge = float(row["edge"])
                actual_total = home_score + away_score
                # We need the market total to determine outcome
                # edge = predicted - market, so market = predicted - edge
                market_total = float(row["predicted_value"]) - edge
                if actual_total > market_total and edge > 0:
                    outcome = "win"  # bet over, went over
                elif actual_total < market_total and edge < 0:
                    outcome = "win"  # bet under, went under
                elif actual_total == market_total:
                    outcome = "push"
                else:
                    outcome = "loss"

        elif market in ("pitcher_k_home", "pitcher_k_away"):
            # K predictions — outcome is the actual K count
            # We can fill this in later when we have prop lines
            # For now, just record what actually happened
            pass

        if outcome:
            execute(
                "UPDATE predictions SET outcome = %s WHERE prediction_id = %s",
                [outcome, pid]
            )
            updated += 1

    print(f"    Updated {updated} outcomes")
    return updated


def backfill_edges():
    """Fill in edges for predictions that had no odds at prediction time."""
    print("\n  Backfilling edges...")

    # Find totals predictions without edge
    no_edge = query("""
        SELECT p.prediction_id, p.game_id, p.predicted_value, p.market
        FROM predictions p
        WHERE p.edge IS NULL AND p.market = 'total' AND p.predicted_value IS NOT NULL
    """)

    if len(no_edge) == 0:
        print("    No edges to backfill")
        return 0

    # Get closing total lines
    odds = query("""
        SELECT game_id, total_line, over_odds, under_odds, sportsbook
        FROM odds
        WHERE market = 'total' AND total_line IS NOT NULL AND is_closing = true
    """)

    book_priority = ["pinnaclesports.com", "bet365", "draftkings", "fanduel", "betmgm", "caesars"]
    best_lines = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        book = r["sportsbook"]
        if gid not in best_lines:
            best_lines[gid] = float(r["total_line"])
        else:
            cur_p = book_priority.index(best_lines.get("_sb", "x")) if best_lines.get("_sb", "x") in book_priority else 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best_lines[gid] = float(r["total_line"])

    updated = 0
    for _, row in no_edge.iterrows():
        gid = row["game_id"]
        if gid in best_lines:
            edge = float(row["predicted_value"]) - best_lines[gid]
            execute(
                "UPDATE predictions SET edge = %s WHERE prediction_id = %s",
                [round(edge, 4), int(row["prediction_id"])]
            )
            updated += 1

    print(f"    Updated {updated} edges")
    return updated


def backfill_moneyline_edges():
    """Fill in ML edges for predictions that had no odds at prediction time."""
    print("\n  Backfilling moneyline edges...")

    no_edge = query("""
        SELECT p.prediction_id, p.game_id, p.predicted_prob
        FROM predictions p
        WHERE p.edge IS NULL AND p.market = 'moneyline' AND p.predicted_prob IS NOT NULL
    """)

    if len(no_edge) == 0:
        print("    No ML edges to backfill")
        return 0

    # Get closing ML implied probs
    odds = query("""
        SELECT game_id, home_implied, sportsbook
        FROM odds
        WHERE market = 'moneyline' AND home_implied IS NOT NULL AND is_closing = true
    """)

    book_priority = ["pinnaclesports.com", "bet365", "draftkings", "fanduel", "betmgm", "caesars"]
    best_implied = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        book = r["sportsbook"]
        if gid not in best_implied:
            best_implied[gid] = float(r["home_implied"])
        else:
            cur_p = 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best_implied[gid] = float(r["home_implied"])

    updated = 0
    for _, row in no_edge.iterrows():
        gid = row["game_id"]
        if gid in best_implied:
            edge = float(row["predicted_prob"]) - best_implied[gid]
            execute(
                "UPDATE predictions SET edge = %s WHERE prediction_id = %s",
                [round(edge, 4), int(row["prediction_id"])]
            )
            updated += 1

    print(f"    Updated {updated} ML edges")
    return updated


def compute_pnl():
    """Compute P&L for flagged bets using flat $100 sizing."""
    print("\n  Computing P&L for flagged bets...")

    # Find bets with outcomes but no P&L
    bets = query("""
        SELECT p.prediction_id, p.game_id, p.market, p.edge,
               p.predicted_value, p.outcome, p.bet_placed
        FROM predictions p
        WHERE p.bet_placed = true AND p.outcome IS NOT NULL AND p.pnl IS NULL
    """)

    if len(bets) == 0:
        print("    No P&L to compute")
        return 0

    BET_AMOUNT = 100.0

    # Get BEST available odds per game per side (line shopping)
    odds = query("""
        SELECT game_id, over_odds, under_odds, sportsbook
        FROM odds
        WHERE market = 'total' AND is_closing = true
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
    """)
    best_odds = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        over = float(r["over_odds"])
        under = float(r["under_odds"])
        if gid not in best_odds:
            best_odds[gid] = {"over_odds": over, "under_odds": under}
        else:
            # Keep best (highest) odds for each side
            if over > best_odds[gid]["over_odds"]:
                best_odds[gid]["over_odds"] = over
            if under > best_odds[gid]["under_odds"]:
                best_odds[gid]["under_odds"] = under

    updated = 0
    for _, row in bets.iterrows():
        gid = row["game_id"]
        outcome = row["outcome"]
        edge = float(row["edge"]) if pd.notna(row["edge"]) else 0

        market_odds = best_odds.get(gid, {"over_odds": -110, "under_odds": -110})

        if edge > 0:  # bet over
            dec_odds = american_to_decimal(market_odds["over_odds"])
        else:  # bet under
            dec_odds = american_to_decimal(market_odds["under_odds"])

        if outcome == "win":
            pnl = BET_AMOUNT * (dec_odds - 1)
        elif outcome == "loss":
            pnl = -BET_AMOUNT
        else:  # push
            pnl = 0

        execute(
            "UPDATE predictions SET bet_amount = %s, bet_odds = %s, pnl = %s WHERE prediction_id = %s",
            [BET_AMOUNT, round(dec_odds, 3), round(pnl, 2), int(row["prediction_id"])]
        )
        updated += 1

    print(f"    Updated {updated} P&L records")
    return updated


def print_bankroll_summary():
    """Print current bankroll status."""
    print(f"\n  {'='*50}")
    print(f"  BANKROLL SUMMARY")
    print(f"  {'='*50}")

    summary = query("""
        SELECT
            COUNT(*) as total_bets,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
            SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COALESCE(SUM(bet_amount), 0) as total_wagered
        FROM predictions
        WHERE bet_placed = true AND model_name LIKE '%_live'
    """)

    if len(summary) > 0:
        r = summary.iloc[0]
        total = int(r["total_bets"] or 0)
        wins = int(r["wins"] or 0)
        losses = int(r["losses"] or 0)
        pending = int(r["pending"] or 0)
        pnl = float(r["total_pnl"] or 0)
        wagered = float(r["total_wagered"] or 0)
        roi = pnl / wagered * 100 if wagered > 0 else 0
        bankroll = 10000 + pnl

        print(f"  Starting bankroll: $10,000")
        print(f"  Total bets:  {total} ({wins}W-{losses}L, {pending} pending)")
        print(f"  Wagered:     ${wagered:,.0f}")
        print(f"  P&L:         ${pnl:+,.2f}")
        print(f"  ROI:         {roi:+.2f}%")
        print(f"  Bankroll:    ${bankroll:,.2f}")
        if wins + losses > 0:
            print(f"  Win rate:    {wins/(wins+losses):.1%}")


def main():
    print(f"\n{'='*60}")
    print("  BACKFILL OUTCOMES & P&L")
    print(f"{'='*60}")

    backfill_edges()
    backfill_moneyline_edges()
    backfill_outcomes()
    compute_pnl()
    print_bankroll_summary()


if __name__ == "__main__":
    main()
