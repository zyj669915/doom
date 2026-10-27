"""HITCOUNT reliability check v3: stationary player, slowly rotate while holding fire.
Enemies are at fixed positions around spawn -> sweeping rotation must hit them.
If hitcount increments during confirmed rotation-firing -> HITCOUNT works.
"""
import sys
sys.path.insert(0, r'D:\fly\doomfly')
sys.path.insert(0, r'D:\fly\experiment\code')
from doom.game import Game

g = Game(seed=41027, scenario='combat_survival')
g.new_episode()
o = g.observation()
h0, a0, k0 = o['hitcount'], o['ammo'], o['kills']
inc = 0
fire_ticks = 0
for t in range(500):
    if g.observation()['finished']:
        g.new_episode()
    g.act({'turn': 6.0, 'forward': 0.0, 'attack': 1})  # rotate right 6deg/frame + always fire
    o = g.observation()
    if o['hitcount'] > h0:
        inc += 1
        h0 = o['hitcount']
print(f'500 ticks (rotate 6deg + fire): kills_delta={o["kills"]-k0} hitcount_delta={h0 - g.observation()["hitcount"] if False else 0}')
# recompute cleanly
o2 = g.observation()
print(f'kills_delta={o2["kills"]-k0} ammo_delta={o2["ammo"]-a0} final_hitcount={o2["hitcount"]} ticks_with_hit_increment={inc}')
print('VERDICT:', 'HITCOUNT_OK' if inc > 0 else 'HITCOUNT_STILL_SUSPICIOUS')
g.close()
