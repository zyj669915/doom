"""HITCOUNT reliability check: play an episode with random-ish shooting, log hitcount/ammo/kills per tick.
Question: does HITCOUNT increment when shooting hits an enemy in combat_survival?
If kills>0 but hitcount stays 0 -> HITCOUNT unreliable -> hit bonus must be redesigned.
"""
import sys
sys.path.insert(0, r'D:\fly\doomfly')
sys.path.insert(0, r'D:\fly\experiment\code')
import numpy as np
from doom.game import Game

g = Game(seed=41027, scenario='combat_survival')
g.new_episode()
prev = g.observation()
seq = []
# aggressive shooting policy: always fire, random turn/forward
for t in range(400):
    if g.observation()['finished']:
        g.new_episode()
    o = g.observation()
    a = {'turn': float(np.random.choice([-6.0, 0.0, 6.0])), 'forward': 0.0, 'attack': 1}
    r = g.act(a)
    o = g.observation()
    seq.append({'t': t, 'ammo': o['ammo'], 'hitcount': o['hitcount'], 'kills': o['kills'],
                'health': o['health'], 'score': o['score']})

ammo_d = seq[-1]['ammo'] - seq[0]['ammo']
hit_d = seq[-1]['hitcount'] - seq[0]['hitcount']
kill_d = seq[-1]['kills'] - seq[0]['kills']
print(f'over 400 ticks (shoot always): ammo_delta={ammo_d} hitcount_delta={hit_d} kills_delta={kill_d}')
# also count ticks where hitcount increments
inc_ticks = sum(1 for i in range(1, len(seq)) if seq[i]['hitcount'] > seq[i-1]['hitcount'])
print(f'ticks with hitcount increment: {inc_ticks}')
# correlation: ammo decreasing (firing) then hitcount increasing soon after
print('sample tail (t, ammo, hitcount, kills):')
for s in seq[-8:]:
    print(f"  t={s['t']} ammo={s['ammo']} hit={s['hitcount']} kills={s['kills']}")
print('VERDICT:', 'HITCOUNT_OK' if (hit_d > 0 or kill_d > 0) else 'HITCOUNT_SUSPICIOUS')
