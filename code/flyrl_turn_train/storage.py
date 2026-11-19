"""独立短任务的只读来源核验与原子 checkpoint；导入本模块不加载脑或游戏。"""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

from .config import PROTOCOL, ROOT, load_config


# 与旧 flyrl.runner.GRAPH_SHA256 相同；不导入绑定重依赖的旧 runner。
GRAPH_SHA256 = 'f4d41f011e97e510761d011634891b1c082f61f91463eb586ee9cf8c6371b1d1'
SCHEMA_VERSION = 1
RESUME_SEMANTICS = ('new_episode; brain/game/history reset; preserve learner/target/optimizer/'
                    'replay/calibration/RNG/cumulative schedule; not exact game restoration')
_SOURCE_FOLDERS = ('experiment/code/flyrl_turn_train', 'experiment/code/flyrl_turn',
                   'experiment/code/flyrl_stage', 'experiment/code/flyrl_next',
                   'experiment/code/flyrl', 'doomfly/doom')
_SOURCE_FILES = ('flyrl_turn_train_cli.py', 'flyrl_turn_cli.py', 'flyrl_stage_cli.py',
                 'corridor_common.py')
_RESUME_MUTABLE = frozenset(('output', 'max_game_tics', 'wall_hours'))


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


def sha256(path):
    """流式散列，不将权威图或 checkpoint 整体读入内存。"""
    with Path(path).open('rb') as stream:
        return _stream_sha256(stream)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _config_sha256(config):
    return hashlib.sha256(_json_bytes(config)).hexdigest()


def write_json(path, value):
    """同目录临时文件加原子替换；序列化失败不会破坏已有 JSON。"""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_json(path, row):
    """追加完整 JSONL 行；先检查有限值，不写入半条序列化结果。"""
    payload = json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
        stream.flush()


def _iwad_path():
    """优先通过包 metadata 定位资产，不实例化 DoomGame。"""
    distribution = importlib.metadata.distribution('vizdoom')
    for entry in sorted(distribution.files or (), key=str):
        if Path(entry).name.lower() == 'freedoom2.wad':
            path = Path(distribution.locate_file(entry))
            if path.is_file():
                return path.resolve()
    # 个别可编辑安装不提供文件清单；调用方必须已完成资源门禁。
    import vizdoom
    path = Path(vizdoom.__file__).resolve().parent / 'freedoom2.wad'
    if not path.is_file():
        raise FileNotFoundError('未找到 ViZDoom freedoom2.wad')
    return path


def provenance(c):
    """绑定真实输入/资产、递归源码、实际解释器和依赖版本；frontend 同样验图。"""
    c = load_config(values=c)
    sources = {}
    for folder in _SOURCE_FOLDERS:
        directory = ROOT / folder
        if not directory.is_dir():
            raise FileNotFoundError(f'来源目录不存在: {directory}')
        paths = sorted(directory.rglob('*.py'))
        if not paths:
            raise ValueError(f'来源目录没有 Python 源码: {directory}')
        for path in paths:
            sources[path.relative_to(ROOT).as_posix()] = sha256(path)
    for name in _SOURCE_FILES:
        path = ROOT / 'experiment/code' / name
        sources[path.relative_to(ROOT).as_posix()] = sha256(path)
    graph = resolve(c['graph']).resolve()
    graph_hash = sha256(graph)
    if graph_hash != GRAPH_SHA256:
        raise ValueError('权威图哈希不匹配；frontend 也必须核验相同图输入')
    iwad = _iwad_path()
    versions = {'python': sys.version, 'executable': sys.executable}
    versions.update({name: importlib.metadata.version(name) for name in
                     ('numpy', 'torch', 'numba', 'vizdoom', 'pandas', 'pyarrow')})
    return {'sources': sources,
            'inputs': {'graph': {'path': str(graph), 'sha256': graph_hash},
                       'iwad': {'path': str(iwad), 'sha256': sha256(iwad)}},
            'versions': versions}


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _validate_provenance(value):
    if not isinstance(value, dict) or set(value) != {'sources', 'inputs', 'versions'}:
        raise ValueError('checkpoint 来源清单结构无效')
    sources, inputs, versions = (value[key] for key in ('sources', 'inputs', 'versions'))
    if (not isinstance(sources, dict) or not sources
            or any(not isinstance(key, str) or not key or '\\' in key
                   or key.startswith('/') or ':' in key or '..' in key.split('/')
                   or not _valid_sha256(digest) for key, digest in sources.items())):
        raise ValueError('checkpoint 源码哈希无效')
    if not isinstance(inputs, dict) or set(inputs) != {'graph', 'iwad'}:
        raise ValueError('checkpoint 输入清单无效')
    for entry in inputs.values():
        if (not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}
                or not isinstance(entry['path'], str) or not entry['path']
                or not _valid_sha256(entry['sha256'])):
            raise ValueError('checkpoint 输入哈希无效')
    required = {'python', 'executable', 'numpy', 'torch', 'numba', 'vizdoom', 'pandas', 'pyarrow'}
    if (not isinstance(versions, dict) or set(versions) != required
            or any(not isinstance(v, str) or not v for v in versions.values())):
        raise ValueError('checkpoint 解释器或依赖版本无效')


def _invalid_constant(value):
    raise ValueError(f'JSON 不允许非有限值: {value}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'JSON 存在重复字段: {key}')
        result[key] = value
    return result


def read_manifest(path):
    """仅解析并校验 JSON；任何 checkpoint tensor/pickle 均不在此读取。"""
    path = Path(path)
    if path.name == 'manifest.json':
        path = path.parent
    if path.name.endswith('.incomplete'):
        raise ValueError('拒绝加载未提交的 incomplete checkpoint')
    with (path / 'manifest.json').open(encoding='utf-8') as stream:
        manifest = json.load(stream, parse_constant=_invalid_constant,
                             object_pairs_hook=_unique_object)
    if (not isinstance(manifest, dict)
            or type(manifest.get('schema_version')) is not int
            or manifest['schema_version'] != SCHEMA_VERSION
            or manifest.get('protocol') != PROTOCOL):
        raise ValueError('不是受支持的独立短任务 checkpoint schema/protocol')
    raw = manifest.get('config')
    if not isinstance(raw, dict) or raw.get('protocol') != PROTOCOL:
        raise ValueError('checkpoint 配置缺失或使用旧协议')
    saved = load_config(values=raw)
    if raw != saved or set(raw) != set(saved):
        raise ValueError('checkpoint 必须保存完整、校验后的配置快照')
    if (not _valid_sha256(manifest.get('config_sha256'))
            or _config_sha256(raw) != manifest['config_sha256']):
        raise ValueError('checkpoint 配置哈希不匹配')
    if not _valid_sha256(manifest.get('state_sha256')):
        raise ValueError('checkpoint state 哈希无效')
    if manifest.get('resume_semantics') != RESUME_SEMANTICS:
        raise ValueError('checkpoint 恢复语义不匹配')
    seconds = manifest.get('checkpoint_write_seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('checkpoint 保存墙钟记账无效')
    _validate_provenance(manifest.get('provenance'))
    return manifest


def save_checkpoint(run_path, c, prov, pipeline, agent, state):
    """完整成功才提交目录并更新 latest；失败保留 incomplete，不覆盖历史。

    checkpoint_write_seconds 计入快照、state 写盘和来源复核，不含其后的
    manifest/目录/latest 最后提交 I/O；恢复只补计这次快照尚未记录的耗时。
    """
    started = time.monotonic()
    c = load_config(values=c)
    prov = deepcopy(prov)
    _validate_provenance(prov)
    state = deepcopy(state)
    if (not isinstance(state, dict) or type(state.get('game_tics')) is not int
            or state['game_tics'] < 0):
        raise ValueError('checkpoint game_tics 必须为非负整数')
    run_path = resolve(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    name = f'checkpoint-{state["game_tics"]:09d}-{uuid.uuid4().hex}'
    temporary, final = run_path / (name + '.incomplete'), run_path / name
    temporary.mkdir(exist_ok=False)
    import torch
    data = {'pipeline': deepcopy(pipeline.state_dict()), 'learner': deepcopy(agent.state_dict()),
            'state': state, 'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state().clone()}
    with (temporary / 'state.pt').open('xb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    state_hash = sha256(temporary / 'state.pt')
    # 保存期间的源码/资产/输入/解释器/版本变化同样必须拒绝，不能只信启动时快照。
    if provenance(c) != prov:
        raise ValueError('保存期间来源、输入或版本发生漂移；保留 incomplete checkpoint')
    manifest = {'schema_version': SCHEMA_VERSION, 'protocol': PROTOCOL, 'config': c,
                'provenance': prov, 'state_sha256': state_hash,
                'config_sha256': _config_sha256(c), 'resume_semantics': RESUME_SEMANTICS,
                'checkpoint_write_seconds': time.monotonic() - started}
    write_json(temporary / 'manifest.json', manifest)
    os.replace(temporary, final)
    try:
        write_json(run_path / 'latest-checkpoint.json', {'checkpoint': str(final)})
    except BaseException:
        # latest 提交失败也回到未提交证据目录，不能留下伪成功的 final。
        os.replace(final, temporary)
        raise
    return final


def load_checkpoint(path, c=None):
    """先拒绝协议/配置/哈希/来源漂移，再使用 CPU weights_only=True 安全读取。"""
    path = Path(path)
    manifest = read_manifest(path)
    saved = load_config(values=manifest['config'])
    if c is not None:
        requested = load_config(values=c)
        if ({k: v for k, v in requested.items() if k not in _RESUME_MUTABLE}
                != {k: v for k, v in saved.items() if k not in _RESUME_MUTABLE}):
            raise ValueError('续训配置与冻结协议不匹配（仅允许 output/max_game_tics/wall_hours 变化）')
    with (path / 'state.pt').open('rb') as stream:
        if _stream_sha256(stream) != manifest['state_sha256']:
            raise ValueError('checkpoint state 哈希漂移；拒绝加载')
        if provenance(saved) != manifest['provenance']:
            raise ValueError('checkpoint 源码、版本或输入漂移；拒绝加载')
        stream.seek(0)
        import torch
        data = torch.load(stream, map_location='cpu', weights_only=True)
    if 'wall_s' in data['state']:
        wall_s = data['state']['wall_s']
        if type(wall_s) not in (int, float) or not math.isfinite(wall_s) or wall_s < 0:
            raise ValueError('checkpoint 累计墙钟无效')
        data['state']['wall_s'] = wall_s + manifest['checkpoint_write_seconds']
    return saved, manifest, data
