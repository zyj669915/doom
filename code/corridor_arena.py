# -*- coding: utf-8 -*-
"""corridor_avoid 场景生成（Windows 重建版）：光流走廊避障。

L 形走廊（3 凸 sector：入口直段 → 右转 90° → 出口段），无敌人/弹药/血包。
纯 UDMF 地图（TEXTMAP+ENDMAP），无 ACS 行为脚本（出口检测与奖励 shaping 均在
Python 侧完成），因此无需 ZDoom ACC 编译器。
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # D:\fly
DOOMFLY = ROOT / 'doomfly'
SCENARIOS = DOOMFLY / 'doom' / 'scenarios'

RULES = {
    'revision': 'corridor-avoid-v1-win',
    'map': 'L-shaped corridor, 3 convex sectors',
    'segments': [
        {'name': 'entry', 'x': [0, 2048], 'y': [0, 256]},
        {'name': 'turn', 'x': [1792, 2048], 'y': [256, 2048]},
        {'name': 'exit', 'x': [1792, 2048], 'y': [2048, 2304]},
    ],
    'player_start': [128, 128],
    'player_start_angle': 0,
    'no_enemies': True,
    'no_ammo': True,
    'no_health': True,
    'wall_texture': 'STARTAN3',
    'floor_texture': 'FLOOR0_5',
    'ceiling_texture': 'CEIL4_1',
    'lightlevel': 224,
    'episode_timeout_tics': 2000,
    'reward': 'shaping in python wrapper: progress-to-exit +0.01*dd, wall-hit -0.15, exit sector +10',
    'observation_boundary': 'pixels only; player position used for reward shaping, never for policy observation',
    'sources': ['https://github.com/ZDoom/acc', 'https://vizdoom.farama.org/'],
}


def map_text():
    """L 形走廊 UDMF。三个凸 sector，转角与出口为双面开口。"""
    lines = ['namespace = "ZDoom";']
    V = {
        'a': (0.0, 0.0),        # 入口西南
        'b': (2048.0, 0.0),     # 入口东南
        'c': (2048.0, 256.0),   # 入口东北
        'd': (0.0, 256.0),      # 入口西北
        'e': (1792.0, 256.0),   # 转角西南
        'f': (2048.0, 256.0),   # = c（共享）
        'g': (2048.0, 2048.0),  # 转角东北
        'h': (1792.0, 2048.0),  # 转角西北
        'i': (2048.0, 2304.0),  # 出口东北
        'j': (1792.0, 2304.0),  # 出口西北
    }
    vid = {k: i for i, k in enumerate(V)}
    for k, (x, y) in V.items():
        lines.append(f'vertex {{ x={x}; y={y}; }}')
    for _ in range(3):
        lines.append('sector { heightfloor=0; heightceiling=192; texturefloor="FLOOR0_5"; textureceiling="CEIL4_1"; lightlevel=224; }')
    side = []

    def add_sd(sec):
        side.append(sec)
        lines.append(f'sidedef {{ sector={sec}; texturemiddle="STARTAN3"; }}')
        return len(side) - 1

    def ld(v1, v2, front, back=None):
        s = f'linedef {{ v1={vid[v1]}; v2={vid[v2]}; sidefront={front};'
        if back is not None:
            s += f' sideback={back};'
        else:
            s += ' blocking=true;'
        lines.append(s + ' }')

    # S0 入口段（sector 0）：a-b-c-d
    ld('a', 'b', add_sd(0))          # 南墙 blocking
    ld('b', 'c', add_sd(0))          # 东墙 blocking
    ld('c', 'e', add_sd(0), add_sd(1))  # 北墙东段：开口接 S1
    ld('e', 'd', add_sd(0))          # 北墙西段 blocking
    ld('d', 'a', add_sd(0))          # 西墙 blocking
    # S1 转角段（sector 1）：e-c-g-h（南边 e-c 已由 c->e 双面定义）
    ld('c', 'g', add_sd(1))          # 东墙 blocking
    ld('g', 'h', add_sd(1), add_sd(2))  # 北墙：开口接 S2
    ld('h', 'e', add_sd(1))          # 西墙 blocking
    # S2 出口段（sector 2）：h-g-i-j（南边 g-h 已定义）
    ld('g', 'i', add_sd(2))          # 东墙 blocking
    ld('i', 'j', add_sd(2))          # 北墙 blocking
    ld('j', 'h', add_sd(2))          # 西墙 blocking
    # 玩家起点
    lines.append('thing { x=128.0; y=128.0; angle=0; type=1; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; single=true; }')
    return '\n'.join(lines).encode()


def write_wad(path, entries):
    offset = 12
    body, directory = bytearray(), bytearray()
    for name, data in entries:
        directory.extend(struct.pack('<ii8s', offset, len(data), name.encode().ljust(8, b'\0')))
        body.extend(data)
        offset += len(data)
    path.write_bytes(struct.pack('<4sii', b'PWAD', len(entries), offset) + body + directory)


def build():
    wad = SCENARIOS / 'corridor_avoid.wad'
    # 纯 UDMF：MAP01 + TEXTMAP + ENDMAP（无 BEHAVIOR，无 ACS 行为脚本）
    write_wad(wad, [('MAP01', b''), ('TEXTMAP', map_text()), ('ENDMAP', b'')])
    (SCENARIOS / 'corridor_avoid.cfg').write_text(
        'doom_scenario_path = corridor_avoid.wad\n'
        'doom_map = map01\n'
        'doom_skill = 3\n'
        'episode_start_time = 10\n'
        'episode_timeout = 2000\n'
        'render_hud = false\n'
        'death_penalty = 1\n'
        'mode = PLAYER\n')
    manifest = {**RULES, 'sha256': {'corridor_avoid.wad': hashlib.sha256(wad.read_bytes()).hexdigest()}}
    (SCENARIOS / 'corridor_avoid.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'wad': str(wad), 'sha256': manifest['sha256']['corridor_avoid.wad']}))


if __name__ == '__main__':
    build()
