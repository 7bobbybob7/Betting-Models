"""
models/mlb/evaluate.py - Generate calibration plots and evaluation reports.

Reads from the predictions table and generates:
1. Calibration plot (predicted vs actual by decile)
2. CLV over time chart
3. Model comparison table
4. Edge distribution histogram

Usage:
    python -m models.mlb.evaluate
    python -m models.mlb.evaluate --save-dir reports/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from db.db import query


def load_predictions():
    """Load all MLB predictions with game data."""
    return query("""
        SELECT
            p.prediction_id, p.game_id, p.model_name, p.market,
            p.predicted_prob, p.edge, p.outcome,
            g.game_date, g.home_score, g.away_score,
            s.year as season,
            ht.name as home_team, at.name as away_team
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        JOIN seasons s ON g.season_id = s.season_id
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE p.market = 'moneyline'
          AND p.model_name LIKE 'mlb%%'
        ORDER BY g.game_date
    """)


def plot_calibration(df, model_name, save_dir):
    """Generate calibration plot for a single model."""
    model_df = df[df["model_name"] == model_name].copy()
    model_df["home_win"] = (model_df["outcome"] == "win").astype(int)

    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_actuals = []
    bin_counts = []

    for i in range(n_bins):
        mask = (model_df["predicted_prob"] >= bins[i]) & (model_df["predicted_prob"] < bins[i + 1])
        subset = model_df[mask]
        if len(subset) >= 10:
            bin_centers.append(subset["predicted_prob"].mean())
            bin_actuals.append(subset["home_win"].mean())
            bin_counts.append(len(subset))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.scatter(bin_centers, bin_actuals, s=[c / 5 for c in bin_counts], c="steelblue", zorder=5)
    ax1.plot(bin_centers, bin_actuals, "o-", color="steelblue", label=model_name)

    # Add count labels
    for x, y, n in zip(bin_centers, bin_actuals, bin_counts):
        ax1.annotate(f"n={n}", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=8, color="gray")

    ax1.set_xlabel("Predicted Probability (Home Win)")
    ax1.set_ylabel("Actual Win Rate")
    ax1.set_title(f"Calibration — {model_name}")
    ax1.legend()
    ax1.set_xlim(0.25, 0.80)
    ax1.set_ylim(0.25, 0.80)
    ax1.grid(True, alpha=0.3)

    # Edge distribution
    edges = model_df["edge"].dropna()
    ax2.hist(edges, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax2.axvline(x=edges.mean(), color="green", linestyle="-", alpha=0.7,
                label=f"Mean CLV: {edges.mean():+.4f}")
    ax2.set_xlabel("Edge (Model Prob - Market Implied)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Edge Distribution — {model_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"calibration_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_clv_over_time(df, save_dir):
    """Plot rolling CLV for each model over time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {"mlb_logreg_v1": "steelblue", "mlb_xgb_v1": "coral"}

    for model_name in df["model_name"].unique():
        model_df = df[df["model_name"] == model_name].copy()
        model_df = model_df.sort_values("game_date")
        edges = model_df["edge"].dropna()
        if len(edges) < 50:
            continue

        # Rolling 200-game CLV
        rolling = edges.rolling(200, min_periods=50).mean()
        dates = model_df.loc[edges.index, "game_date"]

        color = colors.get(model_name, "gray")
        ax.plot(dates.values, rolling.values, label=model_name, color=color, alpha=0.8)

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling 200-Game Mean CLV")
    ax.set_title("CLV Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "clv_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(df, save_dir):
    """Generate side-by-side model comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = sorted(df["model_name"].unique())
    colors = ["steelblue", "coral", "green", "orange"]

    # 1. Calibration comparison
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    for idx, model_name in enumerate(models):
        model_df = df[df["model_name"] == model_name].copy()
        model_df["home_win"] = (model_df["outcome"] == "win").astype(int)

        bins = np.linspace(0, 1, 11)
        centers, actuals = [], []
        for i in range(10):
            mask = (model_df["predicted_prob"] >= bins[i]) & (model_df["predicted_prob"] < bins[i + 1])
            subset = model_df[mask]
            if len(subset) >= 10:
                centers.append(subset["predicted_prob"].mean())
                actuals.append(subset["home_win"].mean())
        if centers:
            ax.plot(centers, actuals, "o-", color=colors[idx % len(colors)], label=model_name, alpha=0.8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Calibration Comparison")
    ax.legend(fontsize=8)
    ax.set_xlim(0.25, 0.80)
    ax.set_ylim(0.25, 0.80)
    ax.grid(True, alpha=0.3)

    # 2. CLV by season
    ax = axes[1]
    for idx, model_name in enumerate(models):
        model_df = df[df["model_name"] == model_name]
        season_clv = model_df.groupby("season")["edge"].mean().dropna()
        ax.bar(season_clv.index + idx * 0.35 - 0.175, season_clv.values,
               width=0.35, color=colors[idx % len(colors)], label=model_name, alpha=0.8)

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean CLV")
    ax.set_title("CLV by Season")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Accuracy by season
    ax = axes[2]
    for idx, model_name in enumerate(models):
        model_df = df[df["model_name"] == model_name].copy()
        model_df["correct"] = (
            ((model_df["predicted_prob"] > 0.5) & (model_df["outcome"] == "win")) |
            ((model_df["predicted_prob"] < 0.5) & (model_df["outcome"] == "loss"))
        ).astype(int)
        season_acc = model_df.groupby("season")["correct"].mean()
        ax.bar(season_acc.index + idx * 0.35 - 0.175, season_acc.values,
               width=0.35, color=colors[idx % len(colors)], label=model_name, alpha=0.8)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Season")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Season")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_summary(df):
    """Print tabular summary of all models."""
    print(f"\n{'='*70}")
    print("  MODEL EVALUATION SUMMARY")
    print(f"{'='*70}")

    for model_name in sorted(df["model_name"].unique()):
        model_df = df[df["model_name"] == model_name].copy()
        model_df["home_win"] = (model_df["outcome"] == "win").astype(int)
        model_df["correct"] = (
            ((model_df["predicted_prob"] > 0.5) & (model_df["outcome"] == "win")) |
            ((model_df["predicted_prob"] < 0.5) & (model_df["outcome"] == "loss"))
        ).astype(int)

        edges = model_df["edge"].dropna()
        n = len(model_df)
        acc = model_df["correct"].mean()
        brier = ((model_df["predicted_prob"] - model_df["home_win"]) ** 2).mean()
        mean_clv = edges.mean() if len(edges) > 0 else None
        clv_pos = (edges > 0).mean() if len(edges) > 0 else None

        print(f"\n  {model_name}:")
        print(f"    Games:       {n:,}")
        print(f"    Accuracy:    {acc:.4f}")
        print(f"    Brier:       {brier:.4f}")
        if mean_clv is not None:
            print(f"    Mean CLV:    {mean_clv:+.4f}")
            print(f"    CLV > 0:     {clv_pos:.1%}")

        # Per-season breakdown
        print(f"    {'Season':<8s} {'Games':>6s} {'Acc':>6s} {'CLV':>8s}")
        for season in sorted(model_df["season"].unique()):
            s = model_df[model_df["season"] == season]
            s_edges = s["edge"].dropna()
            s_acc = s["correct"].mean()
            s_clv = s_edges.mean() if len(s_edges) > 0 else None
            clv_str = f"{s_clv:+.4f}" if s_clv is not None else "   N/A"
            print(f"    {int(season):<8d} {len(s):>6d} {s_acc:>6.3f} {clv_str:>8s}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLB models")
    parser.add_argument("--save-dir", type=str, default="reports",
                        help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nLoading predictions...")
    df = load_predictions()
    print(f"  {len(df)} predictions loaded")
    print(f"  Models: {df['model_name'].unique().tolist()}")
    print(f"  Seasons: {sorted(df['season'].unique().tolist())}")

    # Generate plots
    print(f"\nGenerating plots...")
    for model_name in df["model_name"].unique():
        plot_calibration(df, model_name, args.save_dir)

    plot_clv_over_time(df, args.save_dir)
    plot_model_comparison(df, args.save_dir)

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
