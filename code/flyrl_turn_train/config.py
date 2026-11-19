"""标准库配置校验；在资源准入前安全读取同一份JSON快照。"""
from copy import deepcopy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'malecns-short-turn-learning-v1'
ENV_PROTOCOL = 'malecns-short-turn-v1'
DEFAULTS = {
    'protocol': PROTOCOL, 'environment_protocol': ENV_PROTOCOL, 'schema_version': 1,
    'arm': 'real', 'seed': 41027, 'projection_seed': 23, 'rewire_seed': 17,
    'device': 'cpu', 'frontend': 'agc_motion', 'scale': 1.5, 'bias': 6., 'threshold': -48.,
    'graph': 'doomfly/outputs/doom/malecns_v1/graph.npz',
    'output': 'experiment/runs_turn_train_20261117',
    'max_tics': 280, 'action_repeat': 4, 'gamma_tic': .999, 'max_turn': 6.,
    'max_game_tics': 30000, 'wall_hours': 6., 'eval_wall_hours': 1.,
    'checkpoint_every': 1000, 'calibration_steps': 128,
    'epsilon_end': .1, 'epsilon_decay_steps': 10000,
    'evaluation_points': [10000, 30000], 'ram_gib': 1.5, 'commit_gib': 2.,
    'learner': {'lr': .0003, 'batch_size': 64, 'replay_capacity': 5000,
                'learning_starts': 256, 'target_every': 500, 'max_grad_norm': 10.},
}
FIXED = ('protocol', 'environment_protocol', 'schema_version', 'device', 'frontend',
         'scale', 'bias', 'threshold', 'rewire_seed', 'max_tics', 'action_repeat',
         'gamma_tic', 'max_turn', 'evaluation_points')


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def number(value, name, low, high, *, integer=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not low <= value <= high or (integer and type(value) is not int)):
        raise ValueError(f'{name}类型或范围无效')


def load_config(path=None, *, values=None):
    if path is not None and values is not None:
        raise ValueError('配置路径与快照不能同时提供')
    raw = (json.loads(Path(path).read_text(encoding='utf-8-sig')) if path is not None
           else {} if values is None else values)
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('未知短任务训练配置字段')
    c = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key == 'learner':
            if not isinstance(value, dict) or set(value) - set(DEFAULTS['learner']):
                raise ValueError('未知学习器配置')
            c[key].update(deepcopy(value))
        else:
            c[key] = deepcopy(value)
    for key in FIXED:
        if (type(c[key]) is not type(DEFAULTS[key]) or c[key] != DEFAULTS[key]
                or (isinstance(c[key], list) and any(type(v) is not int for v in c[key]))):
            raise ValueError(f'本适配协议固定字段不可改变：{key}')
    if c['arm'] not in ('real', 'frontend'):
        raise ValueError('仅支持real/frontend，不接旧rewired队列')
    for key in ('seed', 'projection_seed'):
        number(c[key], key, 0, 2**32 - 1, integer=True)
    for key, low, high in (('max_game_tics', 1, 30000), ('checkpoint_every', 1, 10000),
                           ('calibration_steps', 24, 512), ('epsilon_decay_steps', 1, 30000)):
        number(c[key], key, low, high, integer=True)
    for key, low, high in (('wall_hours', 1e-9, 12), ('eval_wall_hours', 1e-9, 2),
                           ('epsilon_end', 0, 1), ('ram_gib', 1.5, 128), ('commit_gib', 2, 256)):
        number(c[key], key, low, high)
    if c['commit_gib'] < c['ram_gib']:
        raise ValueError('commit预算不能小于RAM预算')
    for key in ('graph', 'output'):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError('路径必须为非空字符串')
    for key in ('batch_size', 'replay_capacity', 'learning_starts', 'target_every'):
        number(c['learner'][key], key, 0 if key == 'learning_starts' else 1, 20000, integer=True)
    for key in ('lr', 'max_grad_norm'):
        number(c['learner'][key], key, 1e-12, 100)
    l = c['learner']
    if l['replay_capacity'] > 5000 or max(l['batch_size'], l['learning_starts']) > l['replay_capacity']:
        raise ValueError('replay最多5000，必须容纳batch/warmup')
    return c
