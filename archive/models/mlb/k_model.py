"""
models/mlb/k_model.py - Pitcher strikeout prediction model.

Poisson regression: predicts expected K count for a starter in a specific game.
P(K >= n) for any threshold maps directly to over/under prop lines.

Features:
    - Pitcher Statcast features (whiff rate, chase rate, pitch mix, velo)
    - Pitcher rolling K rates (from box scores)
    - Opposing team K rate
    - Projected IP (from companion outs model or simple estimate)
    - Park and contextual factors

Usage:
    python -m models.mlb.k_model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import poisson, nbinom
import statsmodels.api as sm

from db.db import query
from models.mlb.statcast_features import build_statcast_features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_k_dataset():
    """Build the training dataset for K prediction."""
    print("\nBuilding K prediction dataset...")

    # Load starter game logs with K counts
    starters = query("""
        SELECT
            pg.game_id, pg.player_id as pitcher_id, pg.team_id,
            pg.so as actual_k, pg.ip, pg.pitches,
            g.game_date, g.home_team_id, g.away_team_id,
            g.venue, s.year as season,
            gi.weather_temp, gi.weather_wind
        FROM mlb_pitching_game pg
        JOIN games g ON pg.game_id = g.game_id
        JOIN seasons s ON g.season_id = s.season_id
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        WHERE pg.is_starter = true
          AND pg.so IS NOT NULL
          AND g.sport_id = 2 AND g.status = 'final'
        ORDER BY g.game_date, pg.game_id
    """)
    print(f"  Starter game logs: {len(starters)}")

    # Build Statcast features
    pitcher_feats, team_k_feats = build_statcast_features()

    # Build rolling K features from box scores (non-Statcast)
    print("\n  Building rolling K features from box scores...")
    starters_sorted = starters.sort_values("game_date")
    k_features = {}

    for pid, grp in starters_sorted.groupby("pitcher_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        for i in range(len(grp)):
            row = grp.iloc[i]
            gid = int(row["game_id"])
            prior = grp.iloc[:i]

            if len(prior) < 3:
                k_features[(gid, int(pid))] = {
                    "k_per_start_5": None, "k_per_start_10": None, "k_per_start_szn": None,
                    "ip_per_start_5": None, "ip_per_start_szn": None,
                    "pitches_per_start_5": None,
                }
                continue

            last5 = prior.tail(5)
            last10 = prior.tail(10)
            season_prior = prior[prior["season"] == row["season"]] if "season" in prior.columns else prior

            feat = {
                "k_per_start_5": round(last5["actual_k"].mean(), 2),
                "k_per_start_10": round(last10["actual_k"].mean(), 2) if len(last10) >= 5 else None,
                "k_per_start_szn": round(season_prior["actual_k"].mean(), 2) if len(season_prior) > 0 else None,
                "ip_per_start_5": round(last5["ip"].mean(), 2) if last5["ip"].notna().all() else None,
                "ip_per_start_szn": round(season_prior["ip"].mean(), 2) if len(season_prior) > 0 and season_prior["ip"].notna().all() else None,
                "pitches_per_start_5": round(last5["pitches"].mean(), 1) if last5["pitches"].notna().all() else None,
            }
            k_features[(gid, int(pid))] = feat

    # Assemble dataset
    print("  Assembling dataset...")
    rows = []
    for _, row in starters_sorted.iterrows():
        gid = int(row["game_id"])
        pid = int(row["pitcher_id"])
        tid = int(row["team_id"])

        # Determine opposing team
        if tid == row["home_team_id"]:
            opp_tid = int(row["away_team_id"])
        else:
            opp_tid = int(row["home_team_id"])

        r = {
            "game_id": gid,
            "pitcher_id": pid,
            "game_date": row["game_date"],
            "season": int(row["season"]),
            "actual_k": int(row["actual_k"]),
            "actual_ip": float(row["ip"]) if pd.notna(row["ip"]) else None,
            "weather_temp": row.get("weather_temp"),
            "weather_wind": row.get("weather_wind"),
        }

        # Pitcher Statcast features
        sc = pitcher_feats.get((gid, pid), {})
        r.update(sc)

        # Pitcher rolling K features
        kf = k_features.get((gid, pid), {})
        r.update(kf)

        # Opposing team K rate
        tk = team_k_feats.get((gid, opp_tid), {})
        r.update(tk)

        rows.append(r)

    df = pd.DataFrame(rows)
    print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
K_FEATURES = [
    # Rolling K rates (most predictive)
    "k_per_start_5", "k_per_start_10", "k_per_start_szn",
    # Projected IP (more innings = more K opportunity)
    "ip_per_start_5", "ip_per_start_szn",
    # Pitches per start (workload proxy)
    "pitches_per_start_5",
    # Statcast pitcher features
    "sc_whiff_rate_5", "sc_swstr_rate_5", "sc_chase_rate_5",
    "sc_whiff_rate_10",
    "sc_k_per_start_5", "sc_k_per_start_10",
    "sc_fb_velo_5", "sc_velo_trend",
    # Pitch mix
    "sc_pct_ff", "sc_pct_sl", "sc_pct_ch", "sc_pct_cu",
    # Whiff by pitch type
    "sc_whiff_ff", "sc_whiff_sl", "sc_whiff_ch",
    # Opposing team K rate
    "opp_k_rate_15", "opp_k_rate_30",
    # Contextual
    "weather_temp",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_k_model(df):
    """Train Poisson regression for K prediction."""
    print(f"\n{'='*60}")
    print("  PITCHER K MODEL (Poisson Regression)")
    print(f"{'='*60}")

    # Time-series split
    train = df[df["season"].between(2016, 2022)].copy()
    val = df[df["season"] == 2023].copy()
    test = df[df["season"] == 2024].copy()

    available = [c for c in K_FEATURES if c in df.columns]
    print(f"\n  Features: {len(available)}")

    X_train = train[available].copy()
    y_train = train["actual_k"]
    X_val = val[available].copy()
    y_val = val["actual_k"]
    X_test = test[available].copy()
    y_test = test["actual_k"]

    # Fill NaN with training medians
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    print(f"  Train: {len(X_train)} starts (2016-2022)")
    print(f"  Val:   {len(X_val)} starts (2023)")
    print(f"  Test:  {len(X_test)} starts (2024)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Try multiple alpha values
    best_alpha = None
    best_mae = 999

    print(f"\n  Tuning regularization...")
    for alpha in [0, 0.001, 0.01, 0.1, 1.0]:
        model = PoissonRegressor(alpha=alpha, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"    alpha={alpha:6.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    # Retrain with best alpha
    model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Feature importance
    coefs = pd.Series(model.coef_, index=available).abs().sort_values(ascending=False)
    print(f"\n  Best alpha={best_alpha}")
    print(f"  Top 10 features:")
    for feat, coef in coefs.head(10).items():
        print(f"    {feat:30s} {model.coef_[available.index(feat)]:+.4f}")

    # Evaluate on all splits
    for name, X, y, split_df in [
        ("val (2023)", X_val_scaled, y_val, val),
        ("test (2024)", X_test_scaled, y_test, test),
    ]:
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        print(f"\n  --- {name} ---")
        print(f"  MAE:  {mae:.3f} (target: < 1.5)")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Mean predicted: {preds.mean():.2f}, Mean actual: {y.mean():.2f}")

        # Calibration: for common K lines, check P(over) accuracy
        print(f"  Prop line calibration:")
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            # Model's P(K > line) using Poisson CDF
            p_over = np.array([1 - poisson.cdf(line, mu) for mu in preds])
            actual_over = (y.values > line).astype(float)

            if len(actual_over) > 0:
                model_mean = p_over.mean()
                actual_mean = actual_over.mean()
                n = len(actual_over)
                diff = abs(model_mean - actual_mean)
                flag = " *" if diff > 0.05 else ""
                print(f"    K > {line}: pred={model_mean:.3f} actual={actual_mean:.3f} diff={diff:.3f}{flag}")

        # Residual analysis
        residuals = y.values - preds
        print(f"  Residual stats: mean={residuals.mean():.3f}, std={residuals.std():.3f}")

    # Baseline comparison: always predict mean K
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    print(f"\n  Baseline (predict mean={y_train.mean():.2f}): MAE={baseline_mae:.3f}")
    print(f"  Model improvement over baseline: {(baseline_mae - best_mae) / baseline_mae:.1%}")

    # Save model
    os.makedirs("models/mlb/saved", exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": available,
        "medians": medians,
        "best_alpha": best_alpha,
    }
    with open("models/mlb/saved/k_poisson.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model saved to models/mlb/saved/k_poisson.pkl")

    return model, scaler, available, medians


def train_negbin_model(df):
    """Train Negative Binomial regression — handles overdispersion."""
    print(f"\n{'='*60}")
    print("  PITCHER K MODEL (Negative Binomial)")
    print(f"{'='*60}")

    train = df[df["season"].between(2016, 2022)].copy()
    val = df[df["season"] == 2023].copy()
    test = df[df["season"] == 2024].copy()

    available = [c for c in K_FEATURES if c in df.columns]
    print(f"\n  Features: {len(available)}")

    X_train = train[available].copy()
    y_train = train["actual_k"]
    X_val = val[available].copy()
    y_val = val["actual_k"]
    X_test = test[available].copy()
    y_test = test["actual_k"]

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=available, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=available, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=available, index=X_test.index)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Try different alpha values for NegBin
    best_alpha = None
    best_mae = 999

    print(f"\n  Tuning NegBin alpha (overdispersion)...")
    for alpha in [0.05, 0.1, 0.2, 0.5, 1.0]:
        try:
            nb_model = sm.GLM(
                y_train,
                sm.add_constant(X_train_scaled),
                family=sm.families.NegativeBinomial(alpha=alpha)
            ).fit(disp=False, maxiter=100)

            preds = nb_model.predict(sm.add_constant(X_val_scaled))
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            print(f"    alpha={alpha:.2f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
        except Exception as e:
            print(f"    alpha={alpha:.2f}  FAILED: {e}")

    if best_alpha is None:
        print("  NegBin training failed")
        return None, None, None, None

    # Retrain with best alpha
    nb_model = sm.GLM(
        y_train,
        sm.add_constant(X_train_scaled),
        family=sm.families.NegativeBinomial(alpha=best_alpha)
    ).fit(disp=False, maxiter=100)

    print(f"\n  Best alpha={best_alpha}")

    # Evaluate
    for name, X_s, y, split_name in [
        ("val", X_val_scaled, y_val, "2023"),
        ("test", X_test_scaled, y_test, "2024"),
    ]:
        preds = nb_model.predict(sm.add_constant(X_s))
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        print(f"\n  --- {name} ({split_name}) ---")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Mean predicted: {preds.mean():.2f}, Mean actual: {y.mean():.2f}")

        # Calibration using NegBin distribution
        # NegBin params: mu = predicted mean, alpha = dispersion
        # scipy nbinom: n = 1/alpha, p = 1/(1 + alpha*mu)
        print(f"  Prop line calibration (NegBin CDF):")
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            p_over = []
            for mu in preds:
                n_param = 1.0 / best_alpha
                p_param = n_param / (n_param + mu)
                p_over.append(1 - nbinom.cdf(int(line), n_param, p_param))
            p_over = np.array(p_over)
            actual_over = (y.values > line).astype(float)
            model_mean = p_over.mean()
            actual_mean = actual_over.mean()
            diff = abs(model_mean - actual_mean)
            flag = " *" if diff > 0.05 else ""
            print(f"    K > {line}: pred={model_mean:.3f} actual={actual_mean:.3f} diff={diff:.3f}{flag}")

    # Save
    bundle = {
        "model": nb_model,
        "scaler": scaler,
        "features": available,
        "medians": medians,
        "alpha": best_alpha,
        "model_type": "negbin",
    }
    with open("models/mlb/saved/k_negbin.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model saved to models/mlb/saved/k_negbin.pkl")

    return nb_model, scaler, available, medians


def train_twostage_model(df):
    """Two-stage model: predict IP first, then K = K/9_rate * IP/9."""
    print(f"\n{'='*60}")
    print("  PITCHER K MODEL (Two-Stage: IP -> K)")
    print(f"{'='*60}")

    train = df[df["season"].between(2016, 2022)].copy()
    val = df[df["season"] == 2023].copy()
    test = df[df["season"] == 2024].copy()

    # Stage 1: Predict IP
    ip_features = [
        "ip_per_start_5", "ip_per_start_szn", "pitches_per_start_5",
        "sc_zone_rate_5", "sc_fb_velo_5",
        "opp_k_rate_15",  # weaker opponents = more IP
        "weather_temp",
    ]
    available_ip = [c for c in ip_features if c in df.columns]

    # Stage 2: Predict K/9 rate
    k9_features = [
        "sc_whiff_rate_5", "sc_swstr_rate_5", "sc_chase_rate_5",
        "sc_whiff_rate_10",
        "sc_k_per_start_5", "sc_k_per_start_10",
        "sc_fb_velo_5", "sc_velo_trend",
        "sc_pct_sl", "sc_pct_ch",
        "sc_whiff_sl", "sc_whiff_ch", "sc_whiff_ff",
        "opp_k_rate_15", "opp_k_rate_30",
        "k_per_start_szn",
    ]
    available_k9 = [c for c in k9_features if c in df.columns]

    # Compute actual K/9 for training
    train["actual_k9"] = train["actual_k"] / (train["actual_ip"].replace(0, np.nan)) * 9
    val["actual_k9"] = val["actual_k"] / (val["actual_ip"].replace(0, np.nan)) * 9
    test["actual_k9"] = test["actual_k"] / (test["actual_ip"].replace(0, np.nan)) * 9

    # Remove rows with 0 IP
    train_clean = train[train["actual_ip"] > 0].copy()
    val_clean = val[val["actual_ip"] > 0].copy()
    test_clean = test[test["actual_ip"] > 0].copy()

    medians_ip = train_clean[available_ip].median()
    medians_k9 = train_clean[available_k9].median()

    # --- Stage 1: IP model ---
    print(f"\n  Stage 1: IP prediction ({len(available_ip)} features)")
    from sklearn.linear_model import LinearRegression

    X_ip_train = train_clean[available_ip].fillna(medians_ip)
    y_ip_train = train_clean["actual_ip"]

    ip_model = LinearRegression()
    ip_model.fit(X_ip_train, y_ip_train)

    # --- Stage 2: K/9 model ---
    print(f"  Stage 2: K/9 prediction ({len(available_k9)} features)")
    X_k9_train = train_clean[available_k9].fillna(medians_k9)
    y_k9_train = train_clean["actual_k9"].fillna(train_clean["actual_k9"].median())

    k9_model = LinearRegression()
    k9_model.fit(X_k9_train, y_k9_train)

    # --- Evaluate ---
    for name, split, split_name in [
        ("val", val_clean, "2023"),
        ("test", test_clean, "2024"),
    ]:
        X_ip = split[available_ip].fillna(medians_ip)
        X_k9 = split[available_k9].fillna(medians_k9)

        pred_ip = ip_model.predict(X_ip)
        pred_ip = np.clip(pred_ip, 1.0, 9.0)  # reasonable IP range

        pred_k9 = k9_model.predict(X_k9)
        pred_k9 = np.clip(pred_k9, 0, 20)  # reasonable K/9 range

        pred_k = pred_k9 * pred_ip / 9.0

        actual_k = split["actual_k"].values
        mae = mean_absolute_error(actual_k, pred_k)
        rmse = np.sqrt(mean_squared_error(actual_k, pred_k))
        ip_mae = mean_absolute_error(split["actual_ip"].values, pred_ip)

        print(f"\n  --- {name} ({split_name}) ---")
        print(f"  IP MAE:    {ip_mae:.3f}")
        print(f"  K MAE:     {mae:.3f}")
        print(f"  K RMSE:    {rmse:.3f}")
        print(f"  Mean pred K: {pred_k.mean():.2f}, Mean actual: {actual_k.mean():.2f}")

        # Calibration using Poisson with predicted lambda = pred_k
        print(f"  Prop line calibration:")
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            p_over = np.array([1 - poisson.cdf(line, max(mu, 0.1)) for mu in pred_k])
            actual_over = (actual_k > line).astype(float)
            model_mean = p_over.mean()
            actual_mean = actual_over.mean()
            diff = abs(model_mean - actual_mean)
            flag = " *" if diff > 0.05 else ""
            print(f"    K > {line}: pred={model_mean:.3f} actual={actual_mean:.3f} diff={diff:.3f}{flag}")

    return ip_model, k9_model


def main():
    df = build_k_dataset()

    # Model 1: Poisson (baseline)
    train_k_model(df)

    # Model 2: Negative Binomial (handles overdispersion)
    train_negbin_model(df)

    # Model 3: Two-stage IP -> K
    train_twostage_model(df)

    # Summary comparison
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print("  See individual results above for full metrics.")
    print("  Key comparison: MAE and prop line calibration on 2024 test set.")


if __name__ == "__main__":
    main()
