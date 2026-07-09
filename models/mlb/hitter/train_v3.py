"""
models/mlb/train_v3.py — train + calibrate + save the v3 TB bundle (Attack 3 winner).

v3 = v1 feature stack (108) + 7 advanced-profile features (true pull%, bat tracking,
opposing-catcher framing). Attack 3 result (PRD §10): market+v3 beats market alone on TB
in BOTH time directions (+0.0052 / +0.0121 AUC); features beat identical control 5/5.

Training protocol (differs from v1 by necessity — adv features exist only 2024+):
    base XGB   : 2019-2024  (same window as the passing residual test; 2024 is the only
                 training season where adv features are non-NaN, XGB routes NaN natively)
    isotonic   : 2025       (unseen by base; respects time order)
    live from  : 2026-07    (2026 already served as the residual-test evaluation year)

Saves models/mlb/saved/hitter_tb_xgb_v3.pkl — same bundle keys as v1, so
backtest.load_bundle('tb', 'xgb_v3', ...) and predict_proba() work unchanged.

Usage:  python -m models.mlb.hitter.train_v3
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from models.mlb.hitter.backtest import load_bundle, TARGET_TO_MARKET
from models.mlb.feature_sets import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.feature_sets import ADV_CACHE, ADV_FEATS
from models.mlb.feature_sets import BATCH1
from models.mlb.feature_sets import LUCK

OUT = Path("models/mlb/saved/hitter_tb_xgb_v3.pkl")   # artifact name = "current bundle";
VERSION = "v6"                                        # internal version tracks iterations
ADV_ALL = ADV_FEATS + BATCH1 + LUCK   # batch 1 + batch 3 (luck gap) ACCEPTED on TB gates


def main():
    cfg = TARGET_TO_MARKET['tb']
    label = cfg['label_col']

    from models.mlb.feature_sets import FULL_CACHE
    from models.mlb.feature_sets import build_luck
    from datetime import date as _d
    adv = pd.read_parquet(FULL_CACHE)          # full 2019-2026 coverage (v6 gate, accepted)
    lk = build_luck(_d(2019, 3, 1), _d(2026, 12, 31))
    tr = pd.read_parquet(TRAIN_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    bt = pd.read_parquet(BTEST_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    tr = tr[tr[label].notna()]
    bt = bt[bt[label].notna()]

    base_feats = load_bundle('tb', 'xgb', Path('models/mlb/saved'))['features']
    feats = base_feats + ADV_ALL

    ytr = tr[label].astype(int).values
    print(f"base fit: {len(tr):,} rows (2019-2024), {len(feats)} features")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(tr[feats].values, ytr, verbose=False)

    calib = bt[bt['game_date'].dt.year == 2025]
    ycal = calib[label].astype(int).values
    raw = model.predict_proba(calib[feats].values)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw, ycal)
    cal = calibrator.predict(raw)
    print(f"isotonic on 2025 ({len(calib):,} rows): "
          f"Brier {brier_score_loss(ycal, raw):.5f} -> {brier_score_loss(ycal, cal):.5f}")

    te = bt[bt['game_date'].dt.year == 2026]
    yte = te[label].astype(int).values
    p26 = model.predict_proba(te[feats].values)[:, 1]
    print(f"2026 sanity ({len(te):,} rows): model-alone AUC {roc_auc_score(yte, p26):.4f} "
          f"(blend-vs-market result already established in PRD §10)")

    imp = dict(sorted(zip(feats, model.feature_importances_),
                      key=lambda kv: -kv[1])[:20])
    adv_in_top = [f for f in imp if f in ADV_ALL]
    print(f"adv features in top-20 importance: {adv_in_top}")

    bundle = {
        'target': 'tb', 'cfg': cfg, 'model_type': 'xgb_v3', 'params': XGB_PARAMS,
        'model': model, 'calibrator': calibrator, 'calib_season': 2025,
        'features': feats, 'adv_features': ADV_ALL, 'top_importance': imp,
        'trained_window': '2019-2024', 'version': VERSION,
        'provenance': 'Attack 3 (docs/LEG1_MODEL_V2_PRD.md §10): TB passes both directions',
    }
    with open(OUT, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
