"""
models/wnba/market_audit.py — infer market-id -> stat mapping, then shade-vs-vig audit.

(1) Infer what stat each WNBA market id (390-398) is by matching BettingPros `actual`
    against our player box scores on the same (player, date).
(2) Per bettable book x market: two-sided overround (vig) + blanket-under ROI —
    does the measured ~3pt universal over-shade clear the vig anywhere?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np, pandas as pd
from db.db import query

def dec(o): return np.where(o>0, o/100+1, 100/np.abs(o)+1)

# --- (1) market inference via actual<->boxscore matching ---
props = query("""SELECT prop_date d, market_id m, over_line ln, actual,
    LOWER(player_first_name||' '||player_last_name) fn
    FROM bettingpros_props WHERE book_id=0 AND market_id BETWEEN 390 AND 398
    AND is_scored AND actual IS NOT NULL""")
box = query("""SELECT LOWER(p.full_name) fn, g.game_date d, w.points, w.orb+w.drb rebounds,
    w.assists, w.fg3m threes, w.points+w.orb+w.drb+w.assists pra, w.steals, w.blocks,
    w.points+w.orb+w.drb pr, w.points+w.assists pa
    FROM wnba_player_game w JOIN players p ON w.player_id=p.player_id
    JOIN games g ON w.game_id=g.game_id""")
box['d'] = pd.to_datetime(box['d']).dt.date
props['d'] = pd.to_datetime(props['d']).dt.date
j = props.merge(box, on=['fn','d'], how='inner')
stats = ['points','rebounds','assists','threes','pra','steals','blocks','pr','pa']
print("=== market id -> stat (match rate of `actual` vs box stat) ===")
mapping = {}
for m in sorted(j['m'].unique()):
    sub = j[j['m']==m]
    if len(sub) < 100: continue
    rates = {s: (sub['actual'].astype(float)==sub[s].astype(float)).mean() for s in stats}
    best = max(rates, key=rates.get)
    mapping[m] = best
    print(f"  {m}: {best:>9} ({rates[best]:.0%} match, n={len(sub):,})")

# --- (2) vig + blanket-unders per bettable book x market ---
BOOKS = {60:'Novig',36:'Underdog',63:'Sleeper',37:'PrizePicks',10:'FanDuel'}
print(f"\n=== shade vs vig: blanket ALL-UNDERS ROI ===")
print(f"{'book':>11} {'mkt':>9} {'n':>6} {'vig':>6} {'under_ROI':>10} {'±2SE':>6}")
for bid, bn in BOOKS.items():
    d = query("""SELECT market_id m, over_odds o, under_odds u, over_line ln, actual
        FROM bettingpros_props WHERE book_id=%(b)s AND market_id BETWEEN 390 AND 398
        AND over_odds IS NOT NULL AND under_odds IS NOT NULL AND is_scored AND actual IS NOT NULL""",
        params={'b':bid})
    for m in sorted(d['m'].unique()):
        x = d[d['m']==m].copy()
        if len(x) < 400: continue
        do, du = dec(x['o'].values), dec(x['u'].values)
        vig = (1/do + 1/du - 1).mean()
        won = x['actual'].astype(float).values < x['ln'].values
        profit = np.where(won, du-1, -1.0)
        se = profit.std()/np.sqrt(len(x))
        lbl = mapping.get(m, m)
        flag = ' <<<' if profit.mean() > 0.02 else ''
        print(f"{bn:>11} {str(lbl):>9} {len(x):>6,} {vig*100:>5.1f}% {profit.mean():>+10.4f} {2*se:>6.3f}{flag}")
