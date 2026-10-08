#!/usr/bin/env python3
"""Play the DOOMFLY arenas yourself - no fly brain involved.

Uses the ViZDoom engine already installed in .venv-neural, the same engine
the fly experiment runs on. Default is the exact arena the fly plays
(combat_survival), so you can compare your own survival time with the fly's
(~3.7 s per round in the latest v6 tests).

A polished pygame front-end renders the game (crisp pixel upscale) with a
live HUD: health / armor / ammo / kills / position / round / FPS.

Examples
--------
    # The fly's own arena (combat_survival), human keyboard control
    .venv-neural\\Scripts\\python.exe human_play.py

    # A different ViZDoom scenario map
    .venv-neural\\Scripts\\python.exe human_play.py --scenario health_gathering
    .venv-neural\\Scripts\\python.exe human_play.py --scenario my_way_home
    .venv-neural\\Scripts\\python.exe human_play.py --scenario deadly_corridor
    .venv-neural\\Scripts\\python.exe human_play.py --scenario defend_the_center

    # Full Freedoom 2 campaign level (big real Doom maps, MAP01..MAP32)
    .venv-neural\\Scripts\\python.exe human_play.py --scenario freedoom2 --map MAP02

    # List all available scenarios
    .venv-neural\\Scripts\\python.exe human_play.py --list

    # Short self-playing demo (for previews / screenshots)
    .venv-neural\\Scripts\\python.exe human_play.py --scenario my_way_home --demo 15

Controls (keep the game window focused):
    Left / Right ........... turn
    Up / W ................. move forward
    Down / S ............... move backward
    Ctrl / Space ........... fire
    Esc .................... quit

Notes
-----
- The engine's own PLAYER-mode keyboard handling is NOT reliable on this
  ViZDoom build (verified), so input is read by pygame and fed to the game
  with make_action() - works identically everywhere.
- The fly experiment runs the same engine headless (no window) and drives it
  with neural spike counts - nothing here touches that path.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import vizdoom as vzd

REPO = Path(__file__).resolve().parent
SCENARIO_DIR = REPO / "doom" / "scenarios"          # project's own arenas
VIZDOOM_SCENARIOS = Path(vzd.__file__).parent / "scenarios"
IWAD = Path(vzd.__file__).parent / "freedoom2.wad"  # free replacement data (Freedoom 2)

# (name, description) - scenario configs shipped with the installed ViZDoom
BUILTIN = [
    ("basic", "empty box, move to green armor (simplest)"),
    ("simpler_basic", "even simpler variant"),
    ("defend_the_center", "monsters spawn around you; you turn and shoot"),
    ("defend_the_line", "monsters approach along a line"),
    ("health_gathering", "walk over medkits, avoid damage"),
    ("health_gathering_supreme", "larger health-gathering arena"),
    ("deadly_corridor", "walk a corridor while monsters shoot"),
    ("my_way_home", "maze: find the vest (navigation)"),
    ("predict_position", "shoot moving monsters that cross the room"),
    ("take_cover", "dodge attacks behind cover"),
    ("rocket_basic", "rocket launcher against a monster"),
    ("cig", "CIG multi-objective arena (defend, health, ammo)"),
    ("deathmatch", "deathmatch arena against bots"),
    ("multi_deathmatch", "multiplayer deathmatch map"),
    ("multi_duel", "duel map"),
    ("learning", "minimal learn-to-move map"),
    ("oblige", "larger procedurally-inspired map"),
]

CUSTOM = [
    ("combat_survival", "THE FLY'S ARENA - endless spawns, ammo pickups, die at death"),
]

# ---- visual constants -----------------------------------------------------
BG = (10, 14, 20)
PANEL = (17, 23, 33)
PANEL_EDGE = (38, 48, 66)
ACCENT = (255, 90, 70)        # doom red
ACCENT_DIM = (90, 44, 38)
GREEN = (90, 220, 140)
YELLOW = (255, 205, 66)
BLUE = (110, 170, 255)
TEXT = (220, 228, 238)
DIM = (120, 132, 150)
VIEW_W, VIEW_H = 960, 720     # 1.5x upscale of 640x480
HEADER_H, STATUS_H = 42, 62
WIN_W, WIN_H = VIEW_W + 4, HEADER_H + VIEW_H + STATUS_H + 4


def scenario_cfg(name):
    if name == "combat_survival":
        return SCENARIO_DIR / "combat_survival.cfg", SCENARIO_DIR / "combat_survival.wad"
    if name == "freedoom2":
        return VIZDOOM_SCENARIOS / "freedoom2.cfg", None
    return VIZDOOM_SCENARIOS / f"{name}.cfg", VIZDOOM_SCENARIOS / f"{name}.wad"


def build_game(args):
    cfg, pwad = scenario_cfg(args.scenario)
    if not cfg.exists():
        sys.exit(f"Scenario config not found: {cfg}\nRun with --list to see available scenarios.")
    game = vzd.DoomGame()
    game.load_config(str(cfg))
    game.set_doom_game_path(str(IWAD))
    if pwad is not None:
        game.set_doom_scenario_path(str(pwad))
    if args.map:
        game.set_doom_map(args.map)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)   # pygame renders the frame for us
    game.set_sound_enabled(False)
    game.set_available_buttons([vzd.Button.TURN_LEFT_RIGHT_DELTA,
                                vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,
                                vzd.Button.ATTACK])
    game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, 6)
    game.set_button_max_value(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, 20)
    game.init()
    return game


def gv(game, var):
    try:
        return game.get_game_variable(var)
    except Exception:
        return 0.0


class HUD:
    def __init__(self):
        pygame.font.init()
        self.f_big = pygame.font.SysFont("consolas", 26, bold=True)
        self.f_mid = pygame.font.SysFont("consolas", 20, bold=True)
        self.f_sm = pygame.font.SysFont("microsoftyahei", 16)
        self.f_title = pygame.font.SysFont("consolas", 20, bold=True)
        self.health_color = GREEN
        self.last_h = 100.0

    def header(self, surface, scenario, mapname):
        pygame.draw.rect(surface, PANEL, (0, 0, WIN_W, HEADER_H))  # only header strip
        pygame.draw.line(surface, ACCENT, (0, HEADER_H - 2), (WIN_W, HEADER_H - 2), 2)
        t = self.f_title.render("DOOMFLY", True, ACCENT)
        surface.blit(t, (14, 10))
        t2 = self.f_title.render("HUMAN PLAY", True, TEXT)
        surface.blit(t2, (14 + t.get_width() + 10, 10))
        pill = self.f_sm.render(f" {scenario}{(' ' + mapname) if mapname else ''} ", True,
                                (10, 14, 20))
        pr = pygame.Rect(0, 0, pill.get_width() + 16, pill.get_height() + 8)
        pr.topleft = (14 + t.get_width() + 10 + t2.get_width() + 14, 7)
        pygame.draw.rect(surface, YELLOW, pr, border_radius=4)
        surface.blit(pill, (pr.x + 8, pr.y + 4))
        hint = self.f_sm.render("←→ 转向   ↑/W 前进   ↓/S 后退   Ctrl/Space 开火   Esc 退出",
                                True, DIM)
        surface.blit(hint, (WIN_W - hint.get_width() - 14, 12))

    def status(self, surface, game, round_no, fps, scenario, mapname, died_t):
        top = HEADER_H + VIEW_H + 2
        pygame.draw.rect(surface, PANEL, (0, top, WIN_W, WIN_H - top))  # only status strip
        pygame.draw.line(surface, PANEL_EDGE, (0, top), (WIN_W, top), 2)
        h = gv(game, vzd.GameVariable.HEALTH)
        armor = gv(game, vzd.GameVariable.ARMOR)
        ammo = gv(game, vzd.GameVariable.SELECTED_WEAPON_AMMO)
        kills = gv(game, vzd.GameVariable.KILLCOUNT)
        x = gv(game, vzd.GameVariable.POSITION_X)
        y = gv(game, vzd.GameVariable.POSITION_Y)
        ang = gv(game, vzd.GameVariable.ANGLE)
        if h != self.last_h:
            self.health_color = GREEN if h >= 60 else (YELLOW if h >= 30 else ACCENT)
            self.last_h = h
        boxes = [
            ("HEALTH", f"{h:5.0f}", self.health_color),
            ("ARMOR", f"{armor:4.0f}", BLUE),
            ("AMMO", f"{ammo:5.0f}", YELLOW),
            ("KILLS", f"{kills:5.0f}", TEXT),
            ("POS", f"({x:6.0f},{y:6.0f})", DIM),
            ("ANG", f"{ang:6.1f}°", DIM),
        ]
        cx = 14
        for label, val, color in boxes:
            lb = self.f_sm.render(label, True, DIM)
            vb = self.f_mid.render(val, True, color)
            surface.blit(lb, (cx, top + 8))
            surface.blit(vb, (cx, top + 28))
            cx += max(lb.get_width(), vb.get_width()) + 26
        right = [
            f"ROUND {round_no}",
            f"FPS {fps}",
        ]
        rx = WIN_W - 14
        for item in reversed(right):
            tb = self.f_mid.render(item, True, TEXT)
            rx -= tb.get_width()
            surface.blit(tb, (rx, top + 22))
            rx -= 24
        if died_t and time.time() - died_t < 1.6:
            db = self.f_big.render("YOU DIED", True, ACCENT)
            surface.blit(db, (WIN_W // 2 - db.get_width() // 2, top + 16))


def surface_from_frame(frame):
    h, w = frame.shape[:2]
    if frame.ndim == 3:
        return pygame.image.frombuffer(frame.copy(), (w, h), "RGB")
    return pygame.image.frombuffer(np.repeat(frame[:, :, None], 3, axis=2), (w, h), "RGB")


def play(game, seconds, capture_dir=None, scenario="combat_survival", mapname=None):
    pygame.init()
    window = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"DOOMFLY Human Play - {scenario}"
                               + (f" [{mapname}]" if mapname else ""))
    clock = pygame.time.Clock()
    hud = HUD()
    t0 = time.time()
    last_capture = 0.0
    running = True
    round_no = 0
    died_t = None
    tics = 0

    while running and time.time() - t0 < (seconds if seconds else 1e9):
        if game.is_episode_finished():
            if round_no == 0 or died_t is None:
                pass
            died_t = time.time()
            round_no += 1
            game.new_episode()
        turn, forward, attack = 0.0, 0.0, False
        if seconds:  # demo mode: random wander
            turn = float(random.uniform(-6, 6))
            forward = float(random.uniform(-20, 20))
            attack = random.random() < 0.15
        else:        # human mode: pygame keyboard
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                turn -= 6.0
            if keys[pygame.K_RIGHT]:
                turn += 6.0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                forward += 20.0
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                forward -= 20.0
            attack = bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
                          or keys[pygame.K_SPACE])
        game.make_action([turn, forward, int(attack)], 1)
        tics += 1

        # ---- draw chrome + game ----------------------------------------
        window.fill(BG)
        hud.header(window, scenario, mapname)
        state = game.get_state()
        if state is not None:
            frame = state.screen_buffer
            surf = surface_from_frame(frame)
            scaled = pygame.transform.scale(surf, (VIEW_W, VIEW_H))
            window.blit(scaled, (2, HEADER_H + 2))
            pygame.draw.rect(window, PANEL_EDGE,
                             (0, HEADER_H, WIN_W, HEADER_H + VIEW_H + 2), 2)
            if capture_dir:
                now = time.time()
                if now - last_capture >= 2.0:
                    last_capture = now
                    from PIL import Image
                    out = capture_dir / f"{time.strftime('%H%M%S')}.png"
                    Image.fromarray(frame).save(out)
                    print(f"captured {out}", flush=True)
        hud.status(window, game, round_no, int(clock.get_fps()), scenario, mapname, died_t)
        pygame.display.flip()
        clock.tick(60)

    game.close()
    pygame.quit()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="combat_survival",
                   help="scenario name (default: combat_survival, the fly's arena)")
    p.add_argument("--map", default=None, help="campaign map, e.g. MAP01 (freedoom2 only)")
    p.add_argument("--demo", type=float, default=0,
                   help="self-play for N seconds then exit (preview mode)")
    p.add_argument("--capture-dir", default=None,
                   help="in --demo mode, also save a real game frame every 2 s to this folder")
    p.add_argument("--list", action="store_true", help="list available scenarios")
    args = p.parse_args()

    if args.list:
        print("Custom arenas (this repo):")
        for name, desc in CUSTOM:
            print(f"  {name:22s} {desc}")
        print("ViZDoom built-in scenarios:")
        for name, desc in BUILTIN:
            print(f"  {name:22s} {desc}")
        print("  freedoom2               full Freedoom 2 campaign (--map MAP01..MAP32)")
        return

    print(f"Starting scenario: {args.scenario}"
          + (f" (map {args.map})" if args.map else ""))
    if not args.demo:
        print("Controls: Left/Right turn | Up/W forward | Down/S backward | Ctrl/Space fire | Esc quit")
    game = build_game(args)
    capture = Path(args.capture_dir) if args.capture_dir else None
    if capture:
        capture.mkdir(parents=True, exist_ok=True)
    play(game, args.demo if args.demo else 0, capture_dir=capture,
         scenario=args.scenario, mapname=args.map)


if __name__ == "__main__":
    main()
