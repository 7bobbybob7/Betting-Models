"""
models/mlb/research/circadian_test.py — does the NFL West-coast circadian edge exist in MLB?

Mechanism: athletic performance peaks ~late afternoon body-clock. A PT-team player in an
ET night game (7pm ET = 4pm body clock) is at peak; the ET host is past theirs.
Tests: (1) team level — away win% by body-clock lag x day/night, 2019-2026;
       (2) prop level — market-gap (actual_over - implied) for batters by lag x night,
           2024-2026 Underdog props, split by season for consistency.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import numpy as np, pandas as pd
from db.db import query

# summer UTC offsets (MLB season ~ DST): ET -4, CT -5, MT -6, AZ/PT -7
TZ = {}
for n in ['New York Yankees','New York Mets','Boston Red Sox','Tampa Bay Rays','Toronto Blue Jays',
          'Baltimore Orioles','Philadelphia Phillies','Washington Nationals','Atlanta Braves',
          'Miami Marlins','Pittsburgh Pirates','Cleveland Indians','Cleveland Guardians',
          'Detroit Tigers','Cincinnati Reds']: TZ[n] = -4
for n in ['Chicago Cubs','Chicago White Sox','St. Louis Cardinals','Milwaukee Brewers',
          'Minnesota Twins','Kansas City Royals','Houston Astros','Texas Rangers']: TZ[n] = -5
for n in ['Colorado Rockies']: TZ[n] = -6
for n in ['Arizona Diamondbacks','Los Angeles Dodgers','Los Angeles Angels','San Diego Padres',
          'San Francisco Giants','Oakland Athletics','Athletics','Seattle Mariners']: TZ[n] = -7

ABBR = {'New York Yankees':'NYY','New York Mets':'NYM','Boston Red Sox':'BOS','Tampa Bay Rays':'TB',
 'Toronto Blue Jays':'TOR','Baltimore Orioles':'BAL','Philadelphia Phillies':'PHI',
 'Washington Nationals':'WSH','Atlanta Braves':'ATL','Miami Marlins':'MIA','Pittsburgh Pirates':'PIT',
 'Cleveland Indians':'CLE','Cleveland Guardians':'CLE','Detroit Tigers':'DET','Cincinnati Reds':'CIN',
 'Chicago Cubs':'CHC','Chicago White Sox':'CWS','St. Louis Cardinals':'STL','Milwaukee Brewers':'MIL',
 'Minnesota Twins':'MIN','Kansas City Royals':'KC','Houston Astros':'HOU','Texas Rangers':'TEX',
 'Colorado Rockies':'COL','Arizona Diamondbacks':'ARI','Los Angeles Dodgers':'LAD',
 'Los Angeles Angels':'LAA','San Diego Padres':'SD','San Francisco Giants':'SF',
 'Oakland Athletics':'OAK','Athletics':'ATH','Seattle Mariners':'SEA'}

def dvg(o,u):
    do=np.where(o>0,o/100+1,100/np.abs(o)+1); du=np.where(u>0,u/100+1,100/np.abs(u)+1)
    io,iu=1/do,1/du; return io/(io+iu)

g = query("""SELECT g.game_id, g.game_date, g.game_time, g.home_score, g.away_score,
    th.name home_name, ta.name away_name
    FROM games g JOIN teams th ON g.home_team_id=th.team_id
    JOIN teams ta ON g.away_team_id=ta.team_id
    WHERE g.sport_id=2 AND g.status='final' AND g.game_date>='2019-01-01'
    AND g.game_time IS NOT NULL""")
g['h_tz'] = g['home_name'].map(TZ); g['a_tz'] = g['away_name'].map(TZ)
g = g.dropna(subset=['h_tz','a_tz'])
g['local_hr'] = (pd.to_datetime(g['game_time'], utc=True).dt.hour + g['h_tz']) % 24
g['night'] = g['local_hr'] >= 18
g['lag'] = g['a_tz'] - g['h_tz']          # negative = visitor is WEST of host
g['away_win'] = g['away_score'] > g['home_score']

print("=== TEAM LEVEL: away win% by visitor body-clock lag x day/night (2019-2026) ===")
base = g['away_win'].mean()
print(f"baseline away win%: {base:.3f} (n={len(g):,})")
print(f"{'lag(hrs)':>9} {'time':>6} {'n':>6} {'away_win%':>9} {'edge':>7} {'z':>6}")
for lag in [-3, -2, -1, 0, 1, 2, 3]:
    for night in [True, False]:
        s = g[(g['lag'] == -lag if False else g['lag'] == lag) & (g['night'] == night)]
        if len(s) < 150: continue
        w = s['away_win'].mean(); se = np.sqrt(w*(1-w)/len(s))
        print(f"{lag:>9} {'night' if night else 'day':>6} {len(s):>6,} {w:>9.3f} {w-base:>+7.3f} {(w-base)/se:>+6.2f}")

print("\n=== PROP LEVEL: batter market-gap by lag x night (UD, 2024-2026) ===")
tm = g[['game_date','home_name','away_name','night','lag','h_tz','a_tz']].copy()
rows = []
for _, r in tm.iterrows():
    rows.append((r['game_date'], ABBR[r['away_name']], r['night'], r['lag'], True))
    rows.append((r['game_date'], ABBR[r['home_name']], r['night'], -r['lag'], False))
sched = pd.DataFrame(rows, columns=['prop_date','team','night','body_lag','is_away']).drop_duplicates(['prop_date','team'])

P = query("""SELECT prop_date, player_team team, market_id, over_line ln,
    over_odds o, under_odds u, actual FROM bettingpros_props
    WHERE book_id=36 AND market_id IN (293,403,289)
    AND over_odds IS NOT NULL AND under_odds IS NOT NULL
    AND ABS(over_odds)<=2000 AND ABS(under_odds)<=2000 AND is_scored AND actual IS NOT NULL""")
P['prop_date'] = pd.to_datetime(P['prop_date']).dt.date
sched['prop_date'] = pd.to_datetime(sched['prop_date']).dt.date
J = P.merge(sched, on=['prop_date','team'], how='inner')
J['imp'] = dvg(J['o'].values, J['u'].values)
J['ov'] = (J['actual'].astype(float) > J['ln']).astype(float)
J['yr'] = pd.to_datetime(pd.Series(J['prop_date'])).dt.year
print(f"props matched to schedule: {len(J):,} / {len(P):,}")
print(f"{'segment':>34} {'yr':>5} {'n':>6} {'gap':>7} {'z':>6}")
segs = {
  'WEST batter @ EAST, night (lag<=-2)': (J['body_lag'] <= -2) & J['night'],
  'WEST batter @ EAST, day':             (J['body_lag'] <= -2) & ~J['night'],
  'EAST batter @ WEST, night (lag>=+2)': (J['body_lag'] >= 2) & J['night'],
  'same-zone, night (control)':          (J['body_lag'] == 0) & J['night'],
}
for name, m in segs.items():
    for yr in sorted(J['yr'].unique()):
        s = J[m & (J['yr'] == yr)]
        if len(s) < 150: continue
        gap = s['ov'].mean() - s['imp'].mean()
        se = np.sqrt(s['ov'].mean()*(1-s['ov'].mean())/len(s))
        print(f"{name:>34} {yr:>5} {len(s):>6,} {gap:>+7.3f} {(gap/se if se>0 else 0):>+6.2f}")
