"""Rule teacher v2: hold position + motion-aim + constant scan-fire.
- forward always 0 (don't charge into enemies)
- when motion detected: aim at motion centroid, fire when large/close
- when nothing moves: sweep left/right while firing (scan-fire)
Evaluated over 600 tics (10s), no artificial truncation advantage.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from doom.game import Game

def gray(rgb): return rgb.mean(2).astype(np.float32)

class RuleTeacher:
    def __init__(self):
        self.scan_dir = 1
        self.scan_timer = 45
        self.turn = 0.0
        self.last_fire = 0
        self.prev = None
    def decide(self, rgb):
        """Unified teacher interface: takes a full RGB frame, returns action dict."""
        cur = gray(rgb)
        if self.prev is None:
            self.prev = cur.copy()
        a = self.act(self.prev, cur, 0)
        self.prev = cur
        return a
    def act(self, prev, cur, t):
        d = np.abs(cur - prev)
        mask = d > 12.0
        n = int(mask.sum())
        if n > 2500:
            ys, xs = np.where(mask)
            cx = xs.mean() / cur.shape[1]
            if cx < 0.45: turn = -6.0
            elif cx > 0.55: turn = 6.0
            else: turn = 0.0
            attack = n > 3000
        else:
            self.scan_timer -= 1
            if self.scan_timer <= 0:
                self.scan_dir *= -1
                self.scan_timer = 45
            turn = 6.0 * self.scan_dir
            attack = True  # scan-fire; ammo is plentiful in this arena
        return {'turn': turn, 'forward': 0.0, 'attack': bool(attack)}

def evaluate(seed, tics=600):
    g = Game(seed=seed, scenario='combat_survival')
    rt = RuleTeacher()
    cur = gray(g.pixels())
    t0 = time.perf_counter()
    for t in range(tics):
        if g.game.is_episode_finished(): break
        prev = cur
        a = rt.act(prev, cur, t)
        g.act(a)
        if g.game.is_episode_finished(): break
        cur = gray(g.pixels())
    import vizdoom
    health = int(g.game.get_game_variable(vizdoom.GameVariable.HEALTH))
    kills = int(g.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
    g.close()
    print(f'rule-v2 seed={seed}: tics={t+1} kills={kills} health={health} ({time.perf_counter()-t0:.0f}s)', flush=True)
    return {'tics': t+1, 'kills': kills, 'health': health}

if __name__ == '__main__':
    seeds = [41027, 41028, 41029]
    rows = [evaluate(s) for s in seeds]
    print('\nmean kills:', np.mean([r['kills'] for r in rows]),
          '| mean tics:', np.mean([r['tics'] for r in rows]),
          '| mean health:', np.mean([r['health'] for r in rows]))
    json.dump(rows, open(ROOT/'outputs/doom/teacher-v1/rule-v2-eval.json','w'), indent=1)
