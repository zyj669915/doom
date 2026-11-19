"""真实游戏任务验收；只运行预锁开环策略，不训练、不读取遥测选动作。"""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from flyrl.environment import LayoutConfig
from .config import ROOT, conditions, make_env, manifest
from .environment import PROTOCOL

POLICIES = tuple(f'fixed-{i}' for i in range(6)) + ('periodic-negative', 'periodic-positive',
           'sequence-negative', 'sequence-positive', 'random')


def action_at(policy, tic, rng):
    """仅以策略名、时间与独立随机源决策；不接收RGB、布局或info。"""
    if policy.startswith('fixed-') and policy in POLICIES:
        return int(policy[-1])
    if policy in ('sequence-negative', 'sequence-positive'):
        # 24tic直行，16tic原地转向（名义96度），之后直行；对齐4tic宏步。
        # 两种序列无条件用于每个布局，不按镜像/坐标选择。
        return 3 if tic < 24 or tic >= 40 else (0 if policy == 'sequence-negative' else 4)
    if policy in ('periodic-negative', 'periodic-positive'):
        return 3 if tic % 100 < 76 else (1 if policy == 'periodic-negative' else 5)
    if policy == 'random':
        return int(rng.integers(6))
    raise ValueError('未知验收策略')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def source_record():
    files = [p for p in (ROOT / 'experiment/code/flyrl_turn').glob('*.py')]
    files += [ROOT / ('experiment/code/' + p) for p in
              ('flyrl_turn_cli.py', 'flyrl/environment.py', 'flyrl/__init__.py',
               'flyrl/encoding.py', 'flyrl_stage/resources.py')]
    files += [ROOT / 'doomfly/doom/game.py']
    return {'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'python': sys.version, 'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'vizdoom')},
            'graph_loaded': False, 'learner_loaded': False,
            'policy_input': 'predeclared time/random only; telemetry never selects actions'}


def nominal_conditions():
    return [{'case_id': 'nominal-' + side, 'seed': 94000,
             'layout': asdict(LayoutConfig(layout_id='near_turn', layout_seed=80000, mirror=side,
                         spawn_jitter=0., angle_jitter=0., lightlevel=224))} for side in ('left', 'right')]


def check_stop(path, monitor, started, max_seconds):
    if monitor.stop_requested:
        return 'resource_stop'
    if (path / 'STOP').exists() or (path.parent / 'STOP').exists():
        return 'controlled_stop'
    if time.monotonic() - started >= max_seconds:
        return 'wall_limit'
    return None


def episode(c, path, case, policy, monitor, started, max_seconds, initial_frames):
    env = make_env(c, path, case['layout'], case['seed'])
    rng = np.random.default_rng(95000 + case['seed'])
    try:
        rgb, info = env.reset()
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError('真实游戏必须返回480×640×3 uint8 RGB')
        frame_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        # 初始RGB存为PPM，标准格式无需新增图像依赖；仅作观察审计。
        if case['case_id'] not in initial_frames:
            frame = path / (case['case_id'] + '.ppm')
            frame.write_bytes(b'P6\n640 480\n255\n' + rgb.tobytes())
            initial_frames[case['case_id']] = {'path': str(frame), 'sha256': frame_hash}
        row = {'case_id': case['case_id'], 'layout': case['layout'], 'policy': policy,
               'seed': case['seed'], 'tics': 0, 'return': 0., 'discounted_return': 0.,
               'max_progress': 0., 'initial_progress': info['progress'],
               'initial_remaining': info['initial_remaining'], 'initial_rgb_sha256': frame_hash,
               'action_counts': [0] * 6, 'reward_terms': {}, 'success': False, 'complete': False}
        term = trunc = False
        macro_action = None
        for tic in range(c['max_tics']):
            reason = check_stop(path, monitor, started, max_seconds)
            if reason:
                row['stop'] = reason
                return row
            # 与未来策略一致每4tic决定一次；中间仍逐tic取RGB/计奖。
            if tic % c['action_repeat'] == 0:
                macro_action = action_at(policy, tic, rng)
            rgb, reward, term, trunc, info = env.step(macro_action)
            if not np.isfinite(reward) or (rgb is not None and (rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3))):
                raise ValueError('真实帧或奖励无效')
            row['return'] += reward
            row['discounted_return'] += c['gamma_tic'] ** tic * reward
            row['tics'] += 1
            row['action_counts'][macro_action] += 1
            row['max_progress'] = max(row['max_progress'], info['progress'])
            for key, value in info['reward_terms'].items():
                row['reward_terms'][key] = row['reward_terms'].get(key, 0.) + value
            if term or trunc:
                break
        row.update(success=info['success'], complete=bool(term or trunc), terminated=term,
                   truncated=trunc, stop=info['reason'], bootstrap=info['bootstrap'],
                   final_rgb_available=rgb is not None, wall_hits=info['wall_hits'],
                   final_progress=info['progress'], final_telemetry=info['telemetry'], assets=info['assets'])
        if row['success'] and rgb is not None and case['case_id'].startswith('nominal'):
            (path / (case['case_id'] + '-' + policy + '-success.ppm')).write_bytes(
                b'P6\n640 480\n255\n' + rgb.tobytes())
        return row
    finally:
        env.close()


def accept(c, path, monitor, max_seconds=1200, *, started=None):
    started = time.monotonic() if started is None else started
    write_json(path / 'provenance.json', source_record())
    write_json(path / 'conditions.json', manifest())
    write_json(path / 'plan.json', {'protocol': PROTOCOL, 'policies': POLICIES,
        'task_trivial_threshold': .75, 'max_seconds': max_seconds,
        'scope': 'environment reachability and open-loop negatives only; no neural/learning acceptance',
        'sequence_semantics': 'both sequences applied unconditionally to both nominal sides; existential coverage is not a deployable policy'})
    rows = []
    initial_frames = {}
    # 先检查名义左右是否至少存在一条预锁动作轨迹可达；不按镜像选择序列。
    schedule = [(case, policy) for case in nominal_conditions()
                for policy in ('sequence-negative', 'sequence-positive')]
    schedule += [(case, policy) for policy in POLICIES for case in conditions('train')]
    for case, policy in schedule:
        reason = check_stop(path, monitor, started, max_seconds)
        if reason:
            break
        row = episode(c, path, case, policy, monitor, started, max_seconds, initial_frames)
        rows.append(row)
        with open(path / 'episodes.jsonl', 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        write_json(path / 'status.json', {'completed_episodes': sum(r['complete'] for r in rows),
                   'planned_episodes': len(schedule), 'last_case': case['case_id'], 'policy': policy})
        if not row['complete']:
            reason = row['stop']
            break
    else:
        reason = None
    nominal = {side: any(r['complete'] and r['success'] and r['case_id'] == 'nominal-' + side for r in rows)
               for side in ('left', 'right')}
    scores = {}
    for policy in POLICIES:
        group = [r for r in rows if r['case_id'].startswith('train-') and r['policy'] == policy and r['complete']]
        if len(group) == 12:
            scores[policy] = sum(r['success'] for r in group) / len(group)
    complete = len(rows) == len(schedule) and all(r['complete'] for r in rows)
    trivial = any(score >= .75 for score in scores.values())
    # 最后一次step/close、统计与落盘期间也可能收到STOP或越过预算。
    reason = reason or check_stop(path, monitor, started, max_seconds)
    passed = complete and reason is None and all(nominal.values()) and not trivial
    result = {'protocol': PROTOCOL, 'run': str(path), 'complete': complete, 'passed': passed,
              'stop': reason or ('accepted' if passed else 'acceptance_failed'),
              'nominal_reachability': nominal, 'train_open_loop_scores': scores,
              'planned_episodes': len(schedule), 'completed_episodes': sum(r['complete'] for r in rows),
              'wall_s': time.monotonic() - started, 'frames': initial_frames,
              'unverified': ['full-brain throughput on new task', 'untrained neural policy baseline',
                             'visual necessity', 'learning success', 'dev/confirm frozen evaluations'],
              'training_started': False, 'graph_loaded': False}
    write_json(path / 'result.json', result)
    return result
