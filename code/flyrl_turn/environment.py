"""RGB-only短程转角边界；遥测只计算奖励、成功与审计。"""
from dataclasses import asdict, dataclass
import math
from numbers import Integral

from flyrl.environment import GameEnv, LayoutConfig, potential_shaping

PROTOCOL = 'malecns-short-turn-v1'


@dataclass(frozen=True)
class TurnGoal:
    x_min: float = 1792.
    x_max: float = 2048.
    y_min: float = 512.
    y_max: float = 2304.

    def __post_init__(self):
        if asdict(self) != dict(x_min=1792., x_max=2048., y_min=512., y_max=2304.):
            raise ValueError('v1成功区固定，修改目标须新版本')

    def contains(self, x, y, layout):
        return self.x_min <= x <= self.x_max and self.y_min <= y * layout.sign <= self.y_max


def remaining(x, y, layout):
    """到短门的最近中心线投影路程；仅环境评价使用，左右镜像不变。"""
    y *= layout.sign
    first_x = min(1920., max(128., x))
    second_y = min(512., max(128., y))
    candidates = [(math.hypot(x - first_x, y - 128.), 1920. - first_x + 384.),
                  (math.hypot(x - 1920., y - second_y), 512. - second_y)]
    return min(candidates)[1]


class TurnEnv(GameEnv):
    """复用只读游戏构建边界，覆盖任务终止与potential；每step仍为一tic。"""
    def __init__(self, seed, layout=None, *, max_tics=280, max_turn=6., gamma=.999,
                 rewards=None, run_dir=None, upstream_root=None, game_factory=None):
        layout = layout if isinstance(layout, LayoutConfig) else LayoutConfig(**(layout or {
            'layout_id': 'near_turn', 'spawn_jitter': 32., 'angle_jitter': 20.}))
        if layout.layout_id != 'near_turn' or layout.lightlevel != 224:
            raise ValueError('短程v1限定near_turn出生、光照224')
        if (isinstance(max_tics, bool) or not isinstance(max_tics, Integral) or not 1 <= max_tics <= 280
                or isinstance(max_turn, bool) or max_turn != 6.):
            raise ValueError('v1最多280tic、转向幅度固定6')
        self.goal = TurnGoal()
        self._initial_remaining = None
        super().__init__('corridor', seed, max_tics, layout, max_turn=max_turn, gamma=gamma,
                         rewards=rewards, run_dir=run_dir, upstream_root=upstream_root,
                         game_factory=game_factory)

    def _remaining(self, telemetry):
        return remaining(telemetry.x, telemetry.y, self.layout)

    def _phi(self, telemetry, *, terminal=False):
        # 本局起点Phi=0；不裁剪倒退，保留距离信息，真终止Phi=0。
        if terminal:
            return 0.
        base = self._initial_remaining
        if base is None or base <= 0:
            base = self._remaining(telemetry)
        return self.rewards.potential_scale * (1. - self._remaining(telemetry) / base)

    def _info(self, telemetry, *, success=False, wall_hit=False, reason=None, bootstrap=1):
        info = super()._info(telemetry, success=success, wall_hit=wall_hit, reason=reason, bootstrap=bootstrap)
        distance = self._remaining(telemetry)
        base = self._initial_remaining if self._initial_remaining is not None else distance
        progress = 0. if base <= 0 else 1. - distance / base
        info.update(protocol=PROTOCOL, task='short_turn', goal=asdict(self.goal),
                    initial_remaining=base, remaining=distance,
                    progress=1. if success else progress,
                    progress_semantics='1 - short_goal_remaining / episode_initial_remaining; may be negative')
        return info

    def reset(self, seed=None):
        sx, sy, _ = self.layout.spawn()
        if self.goal.contains(sx, sy, self.layout) or remaining(sx, sy, self.layout) <= 0:
            raise ValueError('出生点不能已达短门')
        self._initial_remaining = None
        rgb, _ = super().reset(seed)
        if not all(math.isfinite(v) for v in asdict(self._telemetry).values()):
            self.close()
            raise FloatingPointError('出生遥测非有限')
        self._initial_remaining = self._remaining(self._telemetry)
        if self._initial_remaining <= 0 or self.goal.contains(self._telemetry.x, self._telemetry.y, self.layout):
            self.close()
            raise ValueError('出生点不能已达短门')
        return rgb, self._info(self._telemetry)

    def step(self, action_id):
        if self._done or self._boundary is None:
            raise RuntimeError('每局step前必须reset；终止后不自动重置')
        action = self.codec.decode(action_id)
        previous = self._telemetry
        self._boundary.act(action)
        self.tics += 1
        engine = self._boundary.game
        finished = bool(engine.is_episode_finished())
        current = self._boundary.telemetry()
        if not all(math.isfinite(v) for v in asdict(current).values()):
            raise FloatingPointError('引擎遥测非有限')
        dead = bool(engine.is_player_dead()) or current.health <= 0
        success = not dead and self.goal.contains(current.x, current.y, self.layout)
        limit = self.tics >= self.max_tics
        term = bool(dead or success or finished)
        trunc = bool(limit and not term)
        reason = ('death' if dead else 'short_goal' if success else
                  'engine_time_limit_no_final_rgb' if finished and limit else
                  'engine_finished_no_final_rgb' if finished else 'time_limit' if trunc else None)
        # 引擎仍在运行时保留本步真实RGB；finished后绝不调用get_state。
        rgb = None if finished else self._boundary.pixels()
        displacement = math.hypot(current.x - previous.x, current.y - previous.y)
        wall = action['forward'] > 0 and displacement < self.rewards.wall_distance
        self.wall_hits += int(wall)
        r = self.rewards
        terms = {'tic': r.tic_cost,
                 'potential': potential_shaping(self._phi(previous), self._phi(current), self.gamma, terminal=term),
                 'goal': r.exit_bonus if success else 0.,
                 'wall': r.wall_penalty if wall else 0.,
                 'death': -r.death_penalty if dead else 0.}
        reward = float(sum(terms.values()))
        if not math.isfinite(reward):
            raise FloatingPointError('短程任务奖励非有限')
        self._telemetry, self._done = current, term or trunc
        info = self._info(current, success=success, wall_hit=wall, reason=reason, bootstrap=0 if term else 1)
        info.update(reward_terms=terms, engine_finished=finished, final_rgb_available=rgb is not None,
                    action=action, displacement=displacement)
        if trunc:
            info['final_observation'] = rgb
        return rgb, reward, term, trunc, info
