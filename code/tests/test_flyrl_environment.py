"""Pure adapter tests: no ViZDoom, checkpoint, teacher or training required."""
import importlib.util
import math
from pathlib import Path
import re
import struct
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "flyrl" / "environment.py"
SPEC = importlib.util.spec_from_file_location("flyrl_environment_under_test", MODULE_PATH)
env = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = env
SPEC.loader.exec_module(env)


class FakeEngine:
    def __init__(self, owner):
        self.owner = owner

    def is_episode_finished(self):
        return self.owner.finished

    def is_player_dead(self):
        return self.owner.dead


class FakeGame:
    def __init__(self, *, seed, samples, finish_at=None, dead_at=None, **kwargs):
        self.seed = seed
        self.samples = samples
        self.finish_at, self.dead_at = finish_at, dead_at
        self.tick, self.pixel_calls = 0, 0
        self.finished = self.dead = self.closed = False
        self.actions = []
        self.assets = {"wad_sha256": "fake-test-wad"}
        self.game = FakeEngine(self)

    def pixels(self):
        if self.finished:
            raise AssertionError("get_state/pixels called after finish")
        self.pixel_calls += 1
        # Distinguishes final tic from reset (a reset frame must never bootstrap).
        return bytes([self.seed % 256, self.tick, 42])

    def telemetry(self):
        return self.samples[min(self.tick, len(self.samples) - 1)]

    def act(self, action):
        self.actions.append(action)
        self.tick += 1
        self.finished = self.tick == self.finish_at
        self.dead = self.tick == self.dead_at
        return 9999.0  # native reward must be ignored

    def close(self):
        self.closed = True


class Factory:
    def __init__(self, samples, **kwargs):
        self.samples, self.kwargs, self.games = samples, kwargs, []

    def __call__(self, **kwargs):
        game = FakeGame(samples=self.samples, **self.kwargs, **kwargs)
        self.games.append(game)
        return game


def sample(x=128, y=128, angle=0, health=100, kills=0, hits=0, ammo=100):
    return env.Telemetry(x, y, angle, health, kills, hits, ammo)


class ActionTests(unittest.TestCase):
    def test_cartesian_spaces_and_roundtrip(self):
        for task, n in (("corridor", 6), ("combat", 12)):
            for max_turn in (6, 15):
                codec = env.ActionCodec(task, max_turn)
                self.assertEqual(codec.n, n)
                actions = [codec.decode(i) for i in range(n)]
                self.assertEqual(len({tuple(a.values()) for a in actions}), n)
                self.assertEqual({a["turn"] for a in actions}, {-max_turn, 0, max_turn})
                self.assertEqual({a["forward"] for a in actions}, {0, 20})
                self.assertEqual({a["attack"] for a in actions}, {0} if task == "corridor" else {0, 1})
                for i, action in enumerate(actions):
                    self.assertEqual(codec.encode(**action), i)

    def test_invalid_actions(self):
        codec = env.ActionCodec("corridor")
        for action in (-1, 6, 2.0, True, "2"):
            with self.assertRaises(ValueError):
                codec.decode(action)
        for turn in (0, -6, 16, math.inf, math.nan):
            with self.assertRaises(ValueError):
                env.ActionCodec("combat", turn)
        with self.assertRaises(ValueError):
            env.ActionCodec("teacher")


class PotentialTests(unittest.TestCase):
    def test_formula_and_terminal_zero(self):
        self.assertEqual(env.potential(100, 100), -5)
        self.assertEqual(env.potential(100, 100, terminal=True), 0)
        self.assertAlmostEqual(env.potential_shaping(-5, -4, .99), 1.04)
        self.assertEqual(env.potential_shaping(-5, -4, .99, terminal=True), 5)

    def test_discounted_telescoping_no_terminal_loophole(self):
        gamma = .99
        for values in ((-5, -5, -5, -5), (-5, -1, -4, -2), (-5, -4, -3, -1)):
            rewards = [env.potential_shaping(values[i], values[i + 1], gamma,
                                            terminal=i == len(values) - 2)
                       for i in range(len(values) - 1)]
            self.assertAlmostEqual(sum(gamma ** i * r for i, r in enumerate(rewards)), 5)

    def test_projection_and_mirror(self):
        left, right = env.LayoutConfig(), env.LayoutConfig(mirror="right")
        self.assertEqual(env.corridor_remaining(128, 128, left), env.CORRIDOR_TOTAL)
        self.assertEqual(env.corridor_remaining(1920, 2048, left), 0)
        self.assertEqual(env.corridor_remaining(1728, 128, left), 2112)
        for x, y in ((128, 128), (1900, 200), (1920, 1700), (1920, 2100)):
            self.assertEqual(env.corridor_remaining(x, y, left),
                             env.corridor_remaining(x, -y, right))


class EpisodeTests(unittest.TestCase):
    def make_env(self, samples, *, task="corridor", max_tics=2000, **fake_kwargs):
        factory = Factory(samples, **fake_kwargs)
        game = env.GameEnv(task, 7, max_tics, run_dir="unused-fake-assets", game_factory=factory)
        self.addCleanup(game.close)
        return game, factory

    def test_one_tic_only_action_drives_game(self):
        game, factory = self.make_env([sample(), sample(angle=6), sample(x=129)])
        with self.assertRaises(RuntimeError):
            game.step(0)
        rgb, info = game.reset()
        self.assertIsInstance(rgb, bytes)
        self.assertNotIsInstance(rgb, dict)
        self.assertEqual(info["tics"], 0)
        self.assertIn("telemetry", info)
        self.assertEqual(factory.games[0].actions, [])
        game.step(game.codec.encode(6, 0))
        self.assertEqual(factory.games[0].tick, 1)
        self.assertEqual(factory.games[0].actions, [{"turn": 6, "forward": 0, "attack": 0}])
        self.assertEqual(factory.games[0].pixel_calls, 2)

    def test_stop_turn_is_not_wall_hit_only_forward_attempt(self):
        game, _ = self.make_env([sample()] * 4)
        game.reset()
        for turn, forward, expected in ((0, 0, False), (6, 0, False), (0, 20, True)):
            _, _, _, _, info = game.step(game.codec.encode(turn, forward))
            self.assertEqual(info["wall_hit"], expected)
            self.assertEqual(info["blocked"], expected)
            self.assertEqual(info["reward_terms"]["wall"], -.02 if expected else 0)
        self.assertEqual(info["wall_hits"], 1)

    def test_wall_displacement_threshold(self):
        game, _ = self.make_env([sample(), sample(x=128.5)])
        game.reset()
        *_, info = game.step(game.codec.encode(0, 20))
        self.assertFalse(info["wall_hit"])

    def test_time_limit_preserves_final_rgb_and_potential(self):
        game, factory = self.make_env([sample(), sample(x=132)], max_tics=1)
        initial, _ = game.reset()
        rgb, reward, terminated, truncated, info = game.step(game.codec.encode(0, 20))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertNotEqual(initial, rgb)
        self.assertEqual(rgb, bytes([7, 1, 42]))
        self.assertIs(info["final_observation"], rgb)
        self.assertEqual(info["bootstrap"], 1)
        self.assertEqual(info["reason"], "time_limit")
        expected = env.potential_shaping(-5, game._phi(sample(x=132)), game.gamma)
        self.assertAlmostEqual(info["reward_terms"]["potential"], expected)
        self.assertAlmostEqual(reward, expected - .001)
        self.assertEqual(len(factory.games), 1)  # no auto-reset
        with self.assertRaises(RuntimeError):
            game.step(0)

    def test_finish_without_rgb_is_explicit_terminal(self):
        for max_tics, reason in ((2000, "engine_finished_no_final_rgb"),
                                 (1, "engine_time_limit_no_final_rgb")):
            game, factory = self.make_env([sample(), sample(x=132)], max_tics=max_tics, finish_at=1)
            game.reset()
            rgb, _, terminated, truncated, info = game.step(0)
            self.assertIsNone(rgb)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["bootstrap"], 0)
            self.assertEqual(info["reason"], reason)
            self.assertFalse(info["final_rgb_available"])
            self.assertEqual(factory.games[0].pixel_calls, 1)
            self.assertEqual(info["reward_terms"]["potential"], 5)
            self.assertNotIn("final_observation", info)

    def test_death_overrides_limit_no_state_after_finish(self):
        game, factory = self.make_env([sample(), sample(health=0)], max_tics=1, finish_at=1, dead_at=1)
        game.reset()
        rgb, _, terminated, truncated, info = game.step(2)
        self.assertIsNone(rgb)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "death")
        self.assertFalse(info["success"])
        self.assertEqual(info["bootstrap"], 0)
        self.assertEqual(factory.games[0].pixel_calls, 1)

    def test_exit_overrides_limit_terminal_phi_zero(self):
        game, _ = self.make_env([sample(x=1920, y=2040), sample(x=1920, y=2049)], max_tics=1)
        game.reset()
        _, _, terminated, truncated, info = game.step(3)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertEqual(info["reason"], "exit")
        self.assertEqual(info["progress"], 1)
        self.assertEqual(info["bootstrap"], 0)
        self.assertEqual(info["reward_terms"]["exit"], 10)
        self.assertAlmostEqual(info["reward_terms"]["potential"], -game._phi(sample(x=1920, y=2040)))

    def test_mirrored_exit(self):
        factory = Factory([sample(x=1920, y=-2040), sample(x=1920, y=-2050)])
        with env.GameEnv("corridor", 7, layout={"mirror": "right"},
                         run_dir="unused", game_factory=factory) as game:
            game.reset()
            *_, info = game.step(3)
            self.assertTrue(info["success"])

    def test_combat_delta_reward_ignores_native_reward(self):
        game, _ = self.make_env([sample(), sample(health=80, kills=2, hits=3)], task="combat")
        game.reset()
        _, reward, terminated, truncated, info = game.step(game.codec.encode(0, 0, 1))
        self.assertAlmostEqual(reward, 20 + .3 - .4 - .001)
        self.assertFalse(terminated or truncated)
        self.assertIsNone(info["progress"])
        self.assertNotIn("potential", info["reward_terms"])
        self.assertEqual(info["action"]["forward"], 0)
        self.assertEqual(info["action"]["attack"], 1)
        self.assertIn("nonbiological", info["reward_semantics"])

    def test_reset_reconstructs_seed_and_closes_previous_game(self):
        game, factory = self.make_env([sample(), sample(x=132)])
        initial, _ = game.reset()
        game.step(3)
        repeated, _ = game.reset()
        self.assertEqual(initial, repeated)
        self.assertTrue(factory.games[0].closed)
        changed, info = game.reset(seed=9)
        self.assertNotEqual(initial, changed)
        self.assertEqual(info["seed"], 9)
        self.assertEqual(len(factory.games), 3)
        self.assertEqual(factory.games[2].actions, [])
        game.close()
        self.assertTrue(factory.games[2].closed)
        with self.assertRaises(RuntimeError):
            game.step(0)

    def test_configuration_validation(self):
        for tics in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                env.GameEnv("corridor", 1, tics)
        with self.assertRaises(ValueError):
            env.GameEnv("combat", 1, layout={"layout_id": "exit"})
        for gamma in (-.1, 1.01, math.nan):
            with self.assertRaises(ValueError):
                env.GameEnv("corridor", 1, gamma=gamma)


class LayoutTests(unittest.TestCase):
    def test_reproducible_curriculum_mirror_light_and_seed(self):
        hashes = set()
        for layout_id in ("entry", "near_turn", "exit"):
            for mirror in ("left", "right"):
                layout = env.LayoutConfig(layout_id=layout_id, mirror=mirror, layout_seed=42)
                wad = env.corridor_wad_bytes(layout)
                self.assertEqual(wad, env.corridor_wad_bytes(layout))
                hashes.add(wad)
                x, y, angle = layout.spawn()
                self.assertTrue(0 <= angle < 360)
                self.assertTrue(64 < x < 1984)
                self.assertTrue(64 < y * layout.sign < 1984)
                changed = env.LayoutConfig(layout_id=layout_id, mirror=mirror, layout_seed=43)
                self.assertNotEqual(wad, env.corridor_wad_bytes(changed))
                darker = env.LayoutConfig(layout_id=layout_id, mirror=mirror, layout_seed=42, lightlevel=128)
                self.assertNotEqual(wad, env.corridor_wad_bytes(darker))
                self.assertEqual(layout.spawn(), darker.spawn())
        self.assertEqual(len(hashes), 6)

    def test_spawn_is_exact_reflection(self):
        for layout_id in ("entry", "near_turn", "exit"):
            left = env.LayoutConfig(layout_id=layout_id, layout_seed=42)
            right = env.LayoutConfig(layout_id=layout_id, layout_seed=42, mirror="right")
            lx, ly, la = left.spawn()
            rx, ry, ra = right.spawn()
            self.assertEqual((lx, -ly, (-la) % 360), (rx, ry, ra))

    def test_wad_container_and_portals(self):
        wad = env.corridor_wad_bytes(env.LayoutConfig())
        magic, count, offset = struct.unpack_from("<4sii", wad)
        self.assertEqual((magic, count), (b"PWAD", 3))
        entries = []
        for i in range(count):
            start, size, name = struct.unpack_from("<ii8s", wad, offset + i * 16)
            entries.append((name.rstrip(b"\0"), wad[start:start + size]))
        self.assertEqual([entry[0] for entry in entries], [b"MAP01", b"TEXTMAP", b"ENDMAP"])
        text = entries[1][1].decode()
        self.assertEqual(text.count("sector {"), 3)
        self.assertEqual(text.count("thing {"), 1)
        self.assertEqual(text.count("twosided=true"), 2)
        sides = re.findall(r"sidedef \{([^}]+)\}", text)
        for front, back in re.findall(r"sidefront=(\d+); sideback=(\d+);", text):
            self.assertNotIn("texturemiddle", sides[int(front)])
            self.assertNotIn("texturemiddle", sides[int(back)])
        self.assertNotIn("script", text.lower())
        self.assertNotIn("special=", text)

    def test_front_sides_face_interior_even_when_mirrored(self):
        for mirror, sign in (("left", 1), ("right", -1)):
            text = env.corridor_map_text(env.LayoutConfig(mirror=mirror)).decode()
            vertices = [tuple(map(float, match)) for match in
                        re.findall(r"vertex \{ x=([-\d.]+); y=([-\d.]+); \}", text)]
            edge = re.search(r"linedef \{ v1=(\d+); v2=(\d+);", text)
            a, b = (vertices[int(index)] for index in edge.groups())
            # First wall faces sector center (128, +/-128) on its right.
            cross = (b[0] - a[0]) * (sign * 128 - a[1]) - (b[1] - a[1]) * (128 - a[0])
            self.assertLess(cross, 0)

    def test_layout_validation(self):
        for kwargs in ({"layout_id": "unknown"}, {"mirror": "up"}, {"spawn_jitter": 49},
                       {"spawn_jitter": math.nan}, {"angle_jitter": 46}, {"lightlevel": 256}):
            with self.assertRaises(ValueError):
                env.LayoutConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
