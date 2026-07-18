"""
models/wnba/train_points.py — production WNBA points bundle (line-conditional v2).

Accepted by prop_model_v2_gate: v2>ctl and v2>market BOTH period-directions on points.
Trains on all data through 2026-06-30 (gate-eval ended June); July+ forward stays honest.
Saves models/wnba/saved/wnba_points_lc.pkl. Scored daily by novig_unders_tracker to rank
the structural unders (blanket +2.6%; model picks which).
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np, pandas as pd
import xgboost as xgb
from models.mlb.feature_sets import XGB_PARAMS
from models.wnba.prop_model_v2_gate import build_dataset, V1, B2, RUNGS, norm
from models.wnba.batch6_gate import add_h2h

OUT = "models/wnba/saved/wnba_points_lc.pkl"

def main():
    os.makedirs("models/wnba/saved", exist_ok=True)
    d = build_dataset()
    d['fn'] = d['full_name'].map(norm)
    d = add_h2h(d[d['points'].notna()], 'points')
    tr = d[d['game_date'] <= '2026-06-30']
    feats = V1 + B2 + ['h2h_resid', 'h2h_n']
    med = tr[feats].median()
    y = tr['points'].astype(float).values
    aug = []
    for r in RUNGS[393]:
        a = tr[feats].fillna(med).copy(); a['line'] = r
        a['y'] = (y > r).astype(int); aug.append(a)
    A = pd.concat(aug, ignore_index=True)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(A[feats + ['line']].values, A['y'].values)
    bundle = {'model': m, 'features': feats, 'medians': med, 'version': 'wnba-v3',
              'trained_through': '2026-06-30', 'market_id': 393,
              'provenance': 'v2 gate + batch6 H2H accepted (h2h>ctl and h2h>mkt both directions)'}
    with open(OUT, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"trained on {len(tr):,} player-games x {len(RUNGS[393])} rungs -> {OUT}")

if __name__ == "__main__":
    main()
