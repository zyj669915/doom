"""Rule-based teacher v2: motion-detection patrol & shoot.
Deterministic, will actively aim at moving enemies. Used as a strong teacher
to replace the degenerate PPO teacher (teacher v1 collapsed to a single argmax).
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from doom.game import Game

def gray(rgb):
    return rgb.mean(2).astype(np.float32)

def rule_action(prev, cur):
    """Motion-based aiming: enemy movement -> turn toward centroid + fire.
    Returns (turn, forward, attack)."""
    d = np.abs(cur - prev)
    h, w = d.shape
    # ignore HUD-ish top rows noise: keep full frame, threshold robustly
    mask = d > 12.0
    n = int(mask.sum())
    attack = n > 60
    if n < 30:
        # nothing moving: patrol (alternate scan is handled outside by callers via a counter)
        return 0.0, 20.0, False
    ys, xs = np.where(mask)
    cx = xs.mean() / w
    if cx < 0.40:
        turn = -6.0
    elif cx > 0.60:
        turn = 6.0
    else:
        turn = 0.0
    return turn, 20.0, attack

def evaluate(seed=41027, tics=210):
    g = Game(seed=seed, scenario='combat_survival')
    cur = gray(g.pixels())
    kills = health = 0
    t0 = time.perf_counter()
    for t in range(tics):
        if g.game.is_episode_finished(): break
        prev = cur
        act = rule_action(prev, cur)
        g.act({'turn': act[0], 'forward': act[1], 'attack': act[2]})
        if g.game.is_episode_finished():
            break
        cur = gray(g.pixels())
    import vizdoom
    health = int(g.game.get_game_variable(vizdoom.GameVariable.HEALTH))
    kills = int(g.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
    g.close()
    print(f'rule-teacher seed={seed}: tics survived={t+1} kills={kills} health={health} ({time.perf_counter()-t0:.0f}s)', flush=True)
    return t+1, kills, health

if __name__ == '__main__':
    seeds = [41027, 41028, 41029]
    rows = [dict(zip(['tics','kills','health'], evaluate(s))) for s in seeds]
    print('\nmean kills:', np.mean([r['kills'] for r in rows]),
          '| mean tics:', np.mean([r['tics'] for r in rows]),
          '| mean health:', np.mean([r['health'] for r in rows]))
    json.dump(rows, open(ROOT/'outputs/doom/teacher-v1/rule-eval.json','w'), indent=1)
