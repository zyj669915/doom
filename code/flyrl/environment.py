"""Pixel-only ViZDoom boundary for NONBIOLOGICAL RL baselines.

No teacher, policy, neural decoder or training lives here. ``reset`` returns
``(uint8 HWC RGB, info)``; ``step`` advances exactly one tic and returns
``(RGB_or_None, reward, terminated, truncated, info)``. Telemetry is ONLY for
reward/evaluation: never concatenate info to policy inputs. The learner must use
this environment's gamma for potential shaping. Time limits retain the actual
final RGB and bootstrap=1; an engine finish without RGB is terminal/bootstrap=0.
Explicit LayoutConfig lists (not episode seeds) should define train/val/test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import math
from numbers import Integral
from pathlib import Path
import random
import struct
import tempfile
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ActionCodec:
    """turn-major, then forward, then attack; forward is NEVER forced on."""
    task: str
    max_turn: float = 6.0

    def __post_init__(self):
        if self.task not in ("corridor", "combat"):
            raise ValueError("task must be corridor or combat")
        if not math.isfinite(self.max_turn) or not 0 < self.max_turn <= 15:
            raise ValueError("max_turn must be in (0, 15]")

    @property
    def n(self) -> int:
        return 6 if self.task == "corridor" else 12

    def decode(self, action_id: int) -> dict[str, float | int]:
        if isinstance(action_id, bool) or not isinstance(action_id, Integral):
            raise ValueError("action_id must be an integer")
        if not 0 <= action_id < self.n:
            raise ValueError(f"action_id must be in [0, {self.n})")
        attacks = 1 if self.task == "corridor" else 2
        movement, attack = divmod(int(action_id), attacks)
        turn, forward = divmod(movement, 2)
        return {"turn": (-self.max_turn, 0.0, self.max_turn)[turn],
                "forward": (0, 20)[forward], "attack": attack}

    def encode(self, turn: float, forward: int, attack: int = 0) -> int:
        for i in range(self.n):
            if self.decode(i) == {"turn": turn, "forward": forward, "attack": attack}:
                return i
        raise ValueError("action is not in this codec")


@dataclass(frozen=True)
class LayoutConfig:
    """Geometry RNG is independent of reset seed; mirror left/right is explicit.

    Base map turns left (+Y); right is its Y reflection. Jitter is applied to the
    player thing before init, never by teleporting/steering the live game.
    """
    layout_id: str = "entry"
    layout_seed: int = 0
    mirror: str = "left"
    spawn_jitter: float = 8.0
    angle_jitter: float = 5.0
    lightlevel: int = 224

    def __post_init__(self):
        if self.layout_id not in ("entry", "near_turn", "exit"):
            raise ValueError("unknown curriculum layout_id")
        if self.mirror not in ("left", "right"):
            raise ValueError("mirror must be left or right")
        if not math.isfinite(self.spawn_jitter) or not 0 <= self.spawn_jitter <= 48:
            raise ValueError("spawn_jitter must be in [0, 48]")
        if not math.isfinite(self.angle_jitter) or not 0 <= self.angle_jitter <= 45:
            raise ValueError("angle_jitter must be in [0, 45]")
        if not isinstance(self.lightlevel, Integral) or not 0 <= self.lightlevel <= 255:
            raise ValueError("lightlevel must be an integer in [0, 255]")

    @property
    def sign(self) -> int:
        return 1 if self.mirror == "left" else -1

    def spawn(self) -> tuple[float, float, int]:
        x, y, angle = {"entry": (128, 128, 0), "near_turn": (1728, 128, 0),
                       "exit": (1920, 1792, 90)}[self.layout_id]
        rng = random.Random(self.layout_seed)
        x += rng.uniform(-self.spawn_jitter, self.spawn_jitter)
        y += rng.uniform(-self.spawn_jitter, self.spawn_jitter)
        angle += rng.uniform(-self.angle_jitter, self.angle_jitter)
        return x, self.sign * y, round(self.sign * angle) % 360


def corridor_map_text(layout: LayoutConfig) -> bytes:
    """Pure UDMF variant of corridor_arena.map_text (no module import/build).

    Same 3 sectors/textures/extent; linedefs face the interior (front is right).
    Mirroring also reverses winding, keeping sidedef/front semantics intact.
    """
    vertices = [(0, 0), (2048, 0), (2048, 256), (0, 256),
                (1792, 256), (2048, 2048), (1792, 2048),
                (2048, 2304), (1792, 2304)]
    lines = ['namespace = "ZDoom";']
    for x, y in vertices:
        lines.append(f"vertex {{ x={x}; y={layout.sign * y}; }}")
    for _ in range(3):
        lines.append('sector { heightfloor=0; heightceiling=192; '
                     'texturefloor="FLOOR0_5"; textureceiling="CEIL4_1"; '
                     f'lightlevel={layout.lightlevel}; }}')
    side_count = 0
    # Source edges run counterclockwise; reverse for ZDoom front/right sides.
    edges = [(0, 1, 0, None), (1, 2, 0, None), (2, 4, 0, 1),
             (4, 3, 0, None), (3, 0, 0, None), (2, 5, 1, None),
             (5, 6, 1, 2), (6, 4, 1, None), (5, 7, 2, None),
             (7, 8, 2, None), (8, 6, 2, None)]
    for v1, v2, front, back in edges:
        if layout.sign == 1:
            v1, v2 = v2, v1
        texture = ' texturemiddle="STARTAN3";' if back is None else ''
        lines.append(f'sidedef {{ sector={front};{texture} }}')
        edge = f"linedef {{ v1={v1}; v2={v2}; sidefront={side_count};"
        side_count += 1
        if back is None:
            edge += " blocking=true;"
        else:
            lines.append(f'sidedef {{ sector={back}; }}')
            edge += f" sideback={side_count}; twosided=true;"
            side_count += 1
        lines.append(edge + " }")
    x, y, angle = layout.spawn()
    lines.append(f"thing {{ x={x:.6f}; y={y:.6f}; angle={angle}; type=1; "
                 "skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; single=true; }")
    return "\n".join(lines).encode("ascii")


def corridor_wad_bytes(layout: LayoutConfig) -> bytes:
    entries = [("MAP01", b""), ("TEXTMAP", corridor_map_text(layout)), ("ENDMAP", b"")]
    body, directory = bytearray(), bytearray()
    for name, data in entries:
        directory.extend(struct.pack("<ii8s", 12 + len(body), len(data), name.encode().ljust(8, b"\0")))
        body.extend(data)
    return struct.pack("<4sii", b"PWAD", len(entries), 12 + len(body)) + body + directory


CORRIDOR_TOTAL = 1792.0 + 1920.0


def corridor_remaining(x: float, y: float, layout: LayoutConfig) -> float:
    """Remaining centerline arclength at nearest projection; mirrored invariant."""
    y *= layout.sign
    # Route (128,128) -> (1920,128) -> (1920,2048), exit gate y=2048.
    first_x = min(1920.0, max(128.0, x))
    second_y = min(2048.0, max(128.0, y))
    candidates = [(math.hypot(x - first_x, y - 128), 1920 - first_x + 1920),
                  (math.hypot(x - 1920, y - second_y), 2048 - second_y)]
    return min(candidates)[1]


def potential(remaining: float, total: float = CORRIDOR_TOTAL, scale: float = 5.0,
              *, terminal: bool = False) -> float:
    if total <= 0 or not math.isfinite(total):
        raise ValueError("total must be finite and positive")
    return 0.0 if terminal else -max(0.0, min(total, remaining)) / total * scale


def potential_shaping(previous: float, following: float, gamma: float,
                      *, terminal: bool = False) -> float:
    return gamma * (0.0 if terminal else following) - previous


@dataclass(frozen=True)
class RewardConfig:
    """Engine reward is ignored. These engineered rewards are nonbiological."""
    potential_scale: float = 5.0
    tic_cost: float = -0.001
    exit_bonus: float = 10.0
    wall_penalty: float = -0.02
    wall_distance: float = 0.5
    kill_bonus: float = 10.0
    hit_bonus: float = 0.1
    damage_cost: float = 0.02
    death_penalty: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("rewards must be finite")
        if self.potential_scale < 0 or self.wall_distance < 0 or self.damage_cost < 0:
            raise ValueError("scale, distance and damage_cost must be nonnegative")


@dataclass(frozen=True)
class Telemetry:
    x: float
    y: float
    angle: float
    health: float
    kills: float = 0.0
    hits: float = 0.0
    ammo: float = 0.0


@dataclass(frozen=True)
class RenderedRGB:
    """Only this stream may be consumed by a policy (never Telemetry)."""
    pixels: Any


def _build_boundary(*, task, seed, max_turn, layout, run_dir, upstream_root):
    """Lazy imports allow pure action/layout/reward tests without ViZDoom."""
    import vizdoom as vzd

    source = upstream_root / "doom" / "game.py"
    spec = importlib.util.spec_from_file_location("_flyrl_upstream_game", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reuse precisely the upstream copy-only pixel and one-tic action methods,
    # but configure before init, without constructing then teleporting Game.
    boundary = module.Game.__new__(module.Game)
    g = boundary.game = vzd.DoomGame()
    try:
        assets_dir = run_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        # ViZDoom otherwise writes _vizdoom.ini to the caller's working tree.
        g.set_doom_config_path(str((run_dir / "vizdoom.ini").resolve()))
        if task == "corridor":
            wad_data = corridor_wad_bytes(layout)
            digest = hashlib.sha256(wad_data).hexdigest()
            wad = assets_dir / f"corridor-{digest}.wad"
            wad.write_bytes(wad_data)
        else:
            wad = upstream_root / "doom" / "scenarios" / "combat_survival.wad"
        # Preserve upstream's 10-tic reset-only spawn warmup: start_time=0
        # produces initial tics that ignore controls. No warmup during step.
        cfg_text = ('doom_map = map01\ndoom_skill = 3\nepisode_start_time = 10\n'
                    'episode_timeout = 0\nrender_hud = false\nmode = PLAYER\n')
        cfg = assets_dir / f"{task}-adapter.cfg"
        cfg.write_text(cfg_text, encoding="ascii")
        iwad = Path(vzd.__file__).parent / "freedoom2.wad"
        g.load_config(str(cfg))
        g.set_doom_scenario_path(str(wad.resolve()))
        g.set_doom_game_path(str(iwad.resolve()))
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_depth_buffer_enabled(False)
        g.set_labels_buffer_enabled(False)
        g.set_automap_buffer_enabled(False)
        g.set_objects_info_enabled(False)
        g.set_sectors_info_enabled(False)
        g.set_available_buttons([vzd.Button.TURN_LEFT_RIGHT_DELTA,
                                 vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, vzd.Button.ATTACK])
        g.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, max_turn)
        g.set_button_max_value(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, 20)
        g.clear_available_game_variables()
        for name in ("POSITION_X", "POSITION_Y", "ANGLE", "HEALTH", "KILLCOUNT", "HITCOUNT", "AMMO2"):
            g.add_available_game_variable(getattr(vzd.GameVariable, name))
        # The wrapper owns timeout, otherwise ViZDoom discards bootstrap RGB.
        g.set_episode_timeout(0)
        g.set_seed(seed)
        g.init()
        boundary.tick = 0
        boundary.episode = 0
        boundary.new_episode()
        boundary.assets = {"vizdoom_version": vzd.__version__, "scenario": task,
                           "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (cfg, wad, iwad)},
                           "wad_sha256": hashlib.sha256(wad.read_bytes()).hexdigest(),
                           "upstream_game_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        boundary.telemetry = lambda: Telemetry(**{
            key: float(g.get_game_variable(getattr(vzd.GameVariable, var)))
            for key, var in (("x", "POSITION_X"), ("y", "POSITION_Y"), ("angle", "ANGLE"),
                             ("health", "HEALTH"), ("kills", "KILLCOUNT"),
                             ("hits", "HITCOUNT"), ("ammo", "AMMO2"))})
        return boundary
    except BaseException:
        g.close()
        raise


class GameEnv:
    """One action == one engine tic; no auto-reset, hidden movement or teacher.

    ``game_factory`` is a testing seam returning an upstream-Game-like object
    with .game, .pixels(), .act(dict), .telemetry(), .assets and .close().
    ``info['bootstrap']`` is 0 for true terminals/missing frames, 1 otherwise.
    A wrapper time limit is truncated=True, terminated=False with final RGB.
    """
    def __init__(self, task: str, seed: int, max_tics: int = 2000,
                 layout: LayoutConfig | Mapping | None = None, *,
                 max_turn: float = 6.0, gamma: float = 0.99,
                 rewards: RewardConfig | Mapping | None = None,
                 run_dir: str | Path | None = None,
                 upstream_root: str | Path | None = None,
                 game_factory: Callable | None = None):
        self.codec = ActionCodec(task, max_turn)
        if isinstance(max_tics, bool) or not isinstance(max_tics, Integral) or max_tics <= 0:
            raise ValueError("max_tics must be a positive integer")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        if task == "combat" and layout is not None:
            raise ValueError("layout variants apply only to corridor, not combat")
        self.task, self.seed, self.max_tics, self.gamma = task, int(seed), int(max_tics), gamma
        self.layout = layout if isinstance(layout, LayoutConfig) else LayoutConfig(**(layout or {}))
        self.rewards = rewards if isinstance(rewards, RewardConfig) else RewardConfig(**(rewards or {}))
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.upstream_root = (Path(upstream_root) if upstream_root is not None
                              else Path(__file__).resolve().parents[3] / "doomfly")
        self._factory = game_factory or _build_boundary
        self._boundary = None
        self._done = True
        self.tics = self.wall_hits = 0
        self._telemetry = None

    @property
    def n_actions(self):
        return self.codec.n

    def _remaining(self, telemetry):
        return corridor_remaining(telemetry.x, telemetry.y, self.layout)

    def _phi(self, telemetry, *, terminal=False):
        return potential(self._remaining(telemetry), scale=self.rewards.potential_scale, terminal=terminal)

    def _info(self, telemetry, *, success=False, wall_hit=False, reason=None, bootstrap=1):
        remaining = self._remaining(telemetry) if self.task == "corridor" else None
        return {"task": self.task, "seed": self.seed, "tics": self.tics,
                "success": success, "wall_hit": wall_hit, "blocked": wall_hit,
                "wall_hits": self.wall_hits,
                "progress": None if remaining is None else 1 - remaining / CORRIDOR_TOTAL,
                "remaining": remaining, "reason": reason, "bootstrap": bootstrap,
                "telemetry": asdict(telemetry), "reward_semantics": "nonbiological engineered RL reward",
                "layout": asdict(self.layout) if self.task == "corridor" else None,
                "assets": self._boundary.assets, "gamma": self.gamma}

    def reset(self, seed: int | None = None):
        self.close()
        if seed is not None:
            self.seed = int(seed)
        if self.run_dir is None:
            self.run_dir = Path(tempfile.mkdtemp(prefix="flyrl-env-"))
        self._boundary = self._factory(task=self.task, seed=self.seed, max_turn=self.codec.max_turn,
                                      layout=self.layout, run_dir=self.run_dir,
                                      upstream_root=self.upstream_root)
        try:
            if self._boundary.game.is_episode_finished():
                raise RuntimeError("engine finished during reset; cannot return an observation")
            self._telemetry = self._boundary.telemetry()
            rgb = RenderedRGB(self._boundary.pixels()).pixels
        except BaseException:
            self.close()
            raise
        self.tics = self.wall_hits = 0
        self._done = False
        return rgb, self._info(self._telemetry)

    def step(self, action_id: int):
        if self._done or self._boundary is None:
            raise RuntimeError("reset is required before step/after episode end")
        action = self.codec.decode(action_id)
        previous = self._telemetry
        self._boundary.act(action)
        self.tics += 1
        engine = self._boundary.game
        finished = engine.is_episode_finished()
        # get_game_variable remains valid at finish; get_state MUST NOT be called.
        current = self._boundary.telemetry()
        dead = bool(engine.is_player_dead()) or current.health <= 0
        success = (self.task == "corridor" and not dead and
                   1792 <= current.x <= 2048 and 2048 <= current.y * self.layout.sign <= 2304)
        limit = self.tics >= self.max_tics
        terminated = dead or success or finished
        truncated = limit and not terminated
        reason = ("death" if dead else "exit" if success else
                  "engine_time_limit_no_final_rgb" if finished and limit else
                  "engine_finished_no_final_rgb" if finished else "time_limit" if limit else None)
        rgb = None if finished else RenderedRGB(self._boundary.pixels()).pixels
        displacement = math.hypot(current.x - previous.x, current.y - previous.y)
        wall_hit = action["forward"] > 0 and displacement < self.rewards.wall_distance
        self.wall_hits += int(wall_hit)
        r = self.rewards
        terms = {"tic": r.tic_cost}
        if self.task == "corridor":
            terms["potential"] = potential_shaping(self._phi(previous), self._phi(current),
                                                   self.gamma, terminal=terminated)
            terms["exit"] = r.exit_bonus if success else 0.0
            terms["wall"] = r.wall_penalty if wall_hit else 0.0
        else:
            terms["kills"] = r.kill_bonus * max(0.0, current.kills - previous.kills)
            terms["hits"] = r.hit_bonus * max(0.0, current.hits - previous.hits)
            terms["damage"] = -r.damage_cost * max(0.0, previous.health - current.health)
        terms["death"] = -r.death_penalty if dead else 0.0
        self._telemetry, self._done = current, bool(terminated or truncated)
        info = self._info(current, success=success, wall_hit=wall_hit, reason=reason,
                          bootstrap=0 if terminated else 1)
        info.update({"reward_terms": terms, "engine_finished": finished,
                     "final_rgb_available": rgb is not None,
                     "action": action, "displacement": displacement})
        if truncated:
            info["final_observation"] = rgb
        return rgb, float(sum(terms.values())), bool(terminated), bool(truncated), info

    def close(self):
        self._done = True
        if self._boundary is not None:
            boundary, self._boundary = self._boundary, None
            boundary.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
