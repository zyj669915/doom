"""短转角条件清单；新任务独立split，不消费旧test，不提供训练命令。"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from flyrl.environment import LayoutConfig, corridor_wad_bytes
from .environment import PROTOCOL

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {'protocol': PROTOCOL, 'max_tics': 280, 'action_repeat': 4,
            'gamma_tic': .999, 'max_turn': 6., 'ram_gib': 1.5, 'commit_gib': 2.,
            'output': 'experiment/runs_turn_20261117'}


def conditions(split):
    if split not in ('train', 'dev', 'confirm'):
        raise ValueError('未知split')
    offset, count, env_offset = {'train': (81000, 12, 91000), 'dev': (82000, 12, 92000),
                                 'confirm': (83000, 24, 93000)}[split]
    # 一对镜像使用相同扰动随机源，左右差别不与扰动混杂；WAD内容仍互斥。
    return [{'case_id': f'{split}-{i:02d}', 'seed': env_offset + i,
             'layout': asdict(LayoutConfig(layout_id='near_turn', layout_seed=offset + i // 2,
                 mirror='left' if i % 2 == 0 else 'right', spawn_jitter=32., angle_jitter=20., lightlevel=224))}
            for i in range(count)]


def load_config(path=None, *, values=None):
    if path is not None and values is not None:
        raise ValueError('配置路径与快照不能同时提供')
    raw = (json.loads(Path(path).read_text(encoding='utf-8-sig')) if path is not None
           else {} if values is None else values)
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('短程配置包含未知字段')
    c = {**DEFAULTS, **raw}
    for key in ('protocol', 'max_tics', 'action_repeat', 'gamma_tic', 'max_turn'):
        if type(c[key]) is not type(DEFAULTS[key]) or c[key] != DEFAULTS[key]:
            raise ValueError(f'v1预锁字段不可改变: {key}')
    for key, floor in (('ram_gib', 1.5), ('commit_gib', 2.)):
        value = c[key]
        if type(value) not in (int, float) or not floor <= value < float('inf'):
            raise ValueError('任务预算不得降低或非有限')
    if c['commit_gib'] < c['ram_gib'] or not isinstance(c['output'], str) or not c['output']:
        raise ValueError('预算或输出路径无效')
    return c


def manifest():
    splits = {s: conditions(s) for s in ('train', 'dev', 'confirm')}
    seen = set()
    for rows in splits.values():
        for row in rows:
            digest = hashlib.sha256(corridor_wad_bytes(LayoutConfig(**row['layout']))).hexdigest()
            if digest in seen:
                raise ValueError('任务split出现实际重复WAD')
            seen.add(digest)
            row['wad_sha256'] = digest
    return {'protocol': PROTOCOL, 'splits': splits, 'scope': 'same L-shaped geometry; condition generalization only'}


def make_env(config, path, layout, seed):
    """供未来独立训练适配器使用；旧stage入口不自动切换到本任务。"""
    from .environment import TurnEnv
    return TurnEnv(seed, layout, max_tics=config['max_tics'], gamma=config['gamma_tic'], run_dir=path)
