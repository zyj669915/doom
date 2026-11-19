"""短任务适配：复用编码/DoubleDQN/宏步，独立训练与冻结dev评估，无队列。"""
import os
from pathlib import Path
import random
import time
from copy import deepcopy

import numpy as np
import torch

from flyrl.environment import LayoutConfig
from flyrl_stage.runner import macro_step as base_macro_step, make_agent
from flyrl_turn.config import conditions, manifest as condition_manifest, make_env as turn_env
from flyrl_turn.acceptance import action_at
from .config import load_config, PROTOCOL
from .encoding import Pipeline
from .storage import (append_json, load_checkpoint, provenance, save_checkpoint, write_json)


def stop_reason(path, monitor, started, seconds, previous_wall=0.):
    if getattr(monitor, 'interrupt_requested', lambda: False)():
        return 'controlled_stop'
    if monitor.stop_requested:
        return 'resource_stop'
    if any((p / 'STOP').exists() for p in (path, path.parent, path.parent.parent)):
        return 'controlled_stop'
    if previous_wall + time.monotonic() - started >= seconds:
        return 'wall_limit'
    return None


def make_env(c, path, case):
    return turn_env(c, path, LayoutConfig(**case['layout']), case['seed'])


def macro_step(env, pipeline, action, c, remaining=None):
    """保留旧宏步折扣逻辑，以只读代理补记中间tic峰值进度。"""
    peak = [0.]
    class AuditBoundary:
        def step(self, action_id):
            result = env.step(action_id)
            peak[0] = max(peak[0], float(result[4].get('progress', 0.)))
            return result
    before = getattr(env, 'tics', 0)
    env._unreturned_macro_tics = 0
    try:
        result = base_macro_step(AuditBoundary(), pipeline, action, c, remaining)
    except BaseException:
        env._unreturned_macro_tics = max(0, getattr(env, 'tics', before) - before)
        raise
    result[-1]['macro_max_progress'] = peak[0]
    return result


def initial_frame(pipeline, rgb, state):
    pipeline.reset()
    pipeline.tic(rgb)
    state['initial_observation_tics'] += 1
    state['brain_ms'] += float(pipeline.last_diagnostic.get('actual_ms', 0.))
    return pipeline.observation()


def audit_reset(path, case, info):
    assets = info.get('assets')
    if assets is not None:
        append_json(path / 'assets.jsonl', {'case_id': case['case_id'], 'seed': case['seed'], 'assets': assets})
    return assets


def episode_row(case, index):
    return {'episode': index, 'case_id': case['case_id'], 'seed': case['seed'],
            'layout': deepcopy(case['layout']), 'tics': 0, 'return': 0.,
            'discounted_return': 0., 'max_progress': 0., 'action_counts': [0] * 6,
            'reward_terms': {}}


def record_step(row, action, reward, info, c):
    k = info['macro_tics']
    row['discounted_return'] += c['gamma_tic'] ** row['tics'] * reward
    row['tics'] += k
    row['return'] += info['raw_reward']
    row['action_counts'][action] += k
    row['max_progress'] = max(row['max_progress'], float(info.get('macro_max_progress', info.get('progress', 0.))))
    for key, value in info['macro_reward_terms'].items():
        row['reward_terms'][key] = row['reward_terms'].get(key, 0.) + value


def finish_row(row, term, trunc, info):
    row.update(success=bool(info['success']), terminated=bool(term), truncated=bool(trunc),
               wall_hits=info['wall_hits'], stop=info.get('reason'), complete=True)
    return row


def run_train(c, path, monitor, *, started=None, resume=None, smoke=False):
    """由门禁后的CLI调用；纯测试可注入轻量环境/编码，不含隐藏调度。"""
    c = load_config(values=c)
    started = time.monotonic() if started is None else started
    random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    prov = provenance(c)
    write_json(path / 'provenance.json', prov)
    write_json(path / 'conditions.json', condition_manifest())
    if resume:
        resume_run = Path(resume).parent
        if any((resume_run / name).exists() for name in ('failure.json', 'failure-progress.json')):
            raise ValueError('异常run的成本/原子状态需要独立审计，拒绝自动续训；仅支持受控边界停止恢复')
    data = load_checkpoint(resume, c)[2] if resume else None
    seconds = min(600., c['wall_hours'] * 3600) if smoke else c['wall_hours'] * 3600
    previous_wall = data['state']['wall_s'] if data else 0.
    if type(previous_wall) not in (int, float) or not np.isfinite(previous_wall) or previous_wall < 0:
        raise ValueError('累计墙钟无效')

    def stopping():
        return stop_reason(path, monitor, started, seconds, previous_wall)

    reason = stopping()
    if reason:
        return {'run': str(path), 'stop': reason, 'complete': False,
                'started': False, 'training_started': False, 'passed': False}
    if data:
        old_state = data['state']
        if (old_state.get('protocol') != PROTOCOL or old_state.get('smoke_only')
                or old_state.get('phase') != 'train'):
            raise ValueError('旧协议/未校准/无优化验收快照不能作为正式续训起点')
        for key in ('game_tics', 'decisions', 'episodes', 'calibration_tics', 'initial_observation_tics'):
            if type(old_state.get(key)) is not int or old_state[key] < 0:
                raise ValueError('续训计数无效')
        if old_state['game_tics'] > c['max_game_tics']:
            raise ValueError('累计目标不能小于checkpoint已有交互')
    pipeline = Pipeline(c, path)
    agent = make_agent(c, pipeline)
    rng = np.random.default_rng(c['seed'])
    state = {'protocol': PROTOCOL, 'game_tics': 0, 'decisions': 0, 'episodes': 0,
             'successful_episodes': 0, 'calibration_tics': 0, 'initial_observation_tics': 0,
             'brain_ms': 0., 'wall_s': 0., 'smoke_only': smoke, 'phase': 'calibration',
             'updates': 0, 'replay_size': 0}
    if data:
        if data['state'].get('protocol') != PROTOCOL or data['state'].get('smoke_only'):
            raise ValueError('旧协议/无优化验收快照不能作为正式续训起点')
        pipeline.load_state_dict(data['pipeline'])
        if not pipeline.calibrated:
            raise ValueError('未完成校准的快照不能续训')
        agent.load_state_dict(data['learner'])
        state = deepcopy(data['state'])
        rng.bit_generator.state = state['numpy_rng']
        random.setstate(data['python_rng'])
        torch.set_rng_state(data['torch_rng'])
        del data
    train_cases = conditions('train')
    env = None
    ep = None
    latest = None
    touched = False

    def save():
        state.update(numpy_rng=deepcopy(rng.bit_generator.state),
                     wall_s=previous_wall + time.monotonic() - started,
                     updates=agent.updates, replay_size=len(agent.replay))
        saved = save_checkpoint(path, c, prov, pipeline, agent, state)
        write_json(path / 'status.json', {'checkpoint': str(saved), **state})
        return saved

    try:
        if not resume and not smoke:
            # 每个训练case固定配额；随机动作不读取布局/遥测，标签只用于采样。
            for index, case in enumerate(train_cases):
                quota = c['calibration_steps'] // len(train_cases) + int(index < c['calibration_steps'] % len(train_cases))
                for _ in range(quota):
                    reason = stopping()
                    if reason:
                        break
                    if env is None or env._done:
                        if env is not None:
                            env.close()
                        env = make_env(c, path, case)
                        rgb, reset_info = env.reset()
                        audit_reset(path, case, reset_info)
                        initial_frame(pipeline, rgb, state)
                        touched = True
                    if stopping():
                        reason = stopping()
                        break
                    _, _, _, _, _, info = macro_step(env, pipeline, int(rng.integers(6)), c)
                    state['calibration_tics'] += info['macro_tics']
                    state['brain_ms'] += info['brain_ms']
                    pipeline.calibration_update()
                if env is not None:
                    env.close()
                    env = None
                reason = reason or stopping()
                if reason:
                    break
            if reason is None:
                pipeline.freeze_calibration()
                write_json(path / 'calibration.json', {
                    'samples': pipeline.calibration_count, 'game_tics': state['calibration_tics'],
                    'scope': 'train-only, all 12 cases, fixed quotas, random 4-tic actions',
                    'constant_dimensions': int(np.count_nonzero(pipeline.sd <= .0100001))})
        if smoke:
            state['phase'] = 'accept'
        elif pipeline.calibrated:
            state['phase'] = 'train'
            if not resume and not stopping():
                latest = save()
                append_json(path / 'evaluation-queue.jsonl', {'game_tics': 0, 'checkpoint': str(latest),
                            'note': 'request only; evaluations never automatically launched'})
        while reason is None and state['game_tics'] < c['max_game_tics']:
            reason = stopping()
            total = state['game_tics'] + state['initial_observation_tics']
            if reason or (smoke and total >= 256):
                reason = reason or 'accept_tic_limit'
                break
            if env is None or env._done:
                if env is not None:
                    env.close()
                reason = stopping()
                if reason:
                    break
                case = train_cases[int(rng.integers(len(train_cases)))]
                env = make_env(c, path, case)
                rgb, reset_info = env.reset()
                audit_reset(path, case, reset_info)
                obs = initial_frame(pipeline, rgb, state)
                touched = True
                ep = episode_row(case, state['episodes'])
            reason = stopping()
            if reason:
                break
            remaining = c['max_game_tics'] - state['game_tics']
            point = next((p for p in c['evaluation_points'] if p > state['game_tics']), c['max_game_tics'])
            remaining = min(remaining, point - state['game_tics'])
            if smoke:
                remaining = min(remaining, 256 - state['game_tics'] - state['initial_observation_tics'])
                if remaining <= 0:
                    reason = 'accept_tic_limit'
                    break
            epsilon = c['epsilon_end'] + (1 - c['epsilon_end']) * max(0., 1 - state['game_tics'] / c['epsilon_decay_steps'])
            action = agent.act(obs, epsilon=epsilon)
            nxt, reward, discount, term, trunc, info = macro_step(env, pipeline, action, c, remaining)
            record_step(ep, action, reward, info, c)
            state['game_tics'] += info['macro_tics']
            state['decisions'] += 1
            state['brain_ms'] += info['brain_ms']
            if not smoke:
                agent.observe(obs, action, reward, nxt, discount, terminated=term, truncated=trunc)
                metrics = agent.update() if not stopping() else None
                if metrics is not None:
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError('学习器指标非有限')
                    append_json(path / 'updates.jsonl', {'game_tics': state['game_tics'], **metrics})
            obs = nxt
            if term or trunc:
                append_json(path / 'episodes.jsonl', finish_row(ep, term, trunc, info))
                state['episodes'] += 1
                state['successful_episodes'] += int(info['success'])
                ep = None
            if state['decisions'] % 100 == 0:
                append_json(path / 'health.jsonl', {'game_tics': state['game_tics'], 'epsilon': epsilon,
                    'vector_std': float(np.std(obs['vector'])),
                    'vector_clip_fraction': float(np.mean(np.abs(obs['vector']) >= 10)),
                    **pipeline.last_diagnostic})
            reason = stopping()
            at_point = state['game_tics'] in c['evaluation_points']
            if not reason and (at_point or state['decisions'] % c['checkpoint_every'] == 0):
                latest = save()
                if at_point and not smoke:
                    append_json(path / 'evaluation-queue.jsonl', {'game_tics': state['game_tics'],
                                'checkpoint': str(latest), 'note': 'not automatically executed'})
    except BaseException as exc:
        # 未完成的optimizer更新不能伪装成安全快照；独立保存成本与失败边界。
        write_json(path / 'failure-progress.json', {
            'type': type(exc).__name__, 'error': str(exc), 'state_completed_boundaries': state,
            'wall_s_observed': previous_wall + time.monotonic() - started,
            'partial_episode': ep, 'last_atomic_checkpoint': str(latest) if latest else None,
            'source_resume': str(resume) if resume else None,
            'unreturned_macro_game_tics': getattr(env, '_unreturned_macro_tics', 0) if env else 0,
            'inflight_transition_or_update_may_be_partial': True,
            'automatic_resume_allowed': False,
            'instruction': '先独立审计失败尝试成本，不把旧快照计数当总执行成本；禁止保存污染更新'})
        raise
    finally:
        if env is not None:
            env.close()
    reason = stopping() or reason
    latest = save()
    reason = stopping() or reason
    accepted = smoke and state['game_tics'] + state['initial_observation_tics'] == 256 and reason in (None, 'accept_tic_limit')
    complete = accepted if smoke else state['game_tics'] == c['max_game_tics'] and reason is None
    return {'run': str(path), 'checkpoint': str(latest), **state, 'previous_wall_s': previous_wall,
            'started': touched, 'training_started': not smoke and state['game_tics'] > 0,
            'optimized': agent.updates > 0, 'complete': bool(complete), 'passed': bool(accepted) if smoke else bool(complete),
            'stop': reason or ('accept_tic_limit' if smoke else 'game_tic_limit'),
            'partial_episode': ep, 'resume_semantics': 'new_episode; brain/game/history reset'}


def run_evaluate(c, path, monitor, checkpoint_path, *, started=None, quick=False,
                 policy='learned', intervention='none'):
    if policy not in ('learned', 'untrained', 'random', 'sequence-negative', 'sequence-positive'):
        raise ValueError('未知冻结评估策略')
    if intervention not in ('none', 'black'):
        raise ValueError('未知输入干预')
    c = load_config(values=c)
    started = time.monotonic() if started is None else started
    seconds = c['eval_wall_hours'] * 3600
    stopping = lambda: stop_reason(path, monitor, started, seconds)
    _, manifest, data = load_checkpoint(checkpoint_path, c)
    if data['state'].get('smoke_only') or data['state'].get('protocol') != PROTOCOL:
        raise ValueError('验收/旧协议快照不能用于正式冻结dev评估')
    if stopping():
        return {'run': str(path), 'stop': stopping(), 'complete': False, 'success_rate': None}
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    pipeline = Pipeline(c, path)
    pipeline.load_state_dict(data['pipeline'])
    if not pipeline.calibrated:
        raise ValueError('冻结评估要求完整train-only校准')
    pipeline.set_intervention(intervention)
    agent = make_agent(c, pipeline)
    if policy == 'learned':
        agent.load_state_dict(data['learner'], restore_optimizer=False, restore_replay=False, restore_rng=False)
    training_tics = data['state']['game_tics']
    del data
    write_json(path / 'provenance.json', manifest['provenance'])
    write_json(path / 'conditions.json', condition_manifest())
    cases = conditions('dev')[:4] if quick else conditions('dev')
    rows = []
    partial = None
    counters = {'initial_observation_tics': 0, 'brain_ms': 0., 'evaluation_game_tics': 0}
    reason = None
    for case in cases:
        reason = stopping()
        if reason:
            break
        env = make_env(c, path, case)
        row = episode_row(case, len(rows))
        rng = np.random.default_rng(c['seed'] + 999 + case['seed'])
        try:
            rgb, reset_info = env.reset()
            row['assets'] = audit_reset(path, case, reset_info)
            obs = initial_frame(pipeline, rgb, counters)
            while not (reason := stopping()):
                action = (agent.act(obs, epsilon=0.) if policy in ('learned', 'untrained')
                          else action_at(policy, row['tics'], rng))
                obs, reward, _, term, trunc, info = macro_step(env, pipeline, action, c)
                record_step(row, action, reward, info, c)
                counters['brain_ms'] += info['brain_ms']
                counters['evaluation_game_tics'] += info['macro_tics']
                if term or trunc:
                    rows.append(finish_row(row, term, trunc, info))
                    append_json(path / 'episodes.jsonl', row)
                    break
        finally:
            env.close()
        reason = stopping() or reason
        if reason:
            if not row.get('complete'):
                partial = row
                write_json(path / 'partial-episode.json', row)
            break
    if provenance(c) != manifest['provenance']:
        reason = 'source_drift'
    reason = stopping() or reason
    complete = len(rows) == len(cases) and reason is None
    return {'run': str(path), 'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': manifest['state_sha256'], 'split': 'dev', 'quick': quick,
            'policy': policy, 'intervention': intervention, 'frozen': True,
            'training_game_tics': training_tics, 'complete': complete,
            'episodes': rows, 'partial_episode': partial, 'planned_episodes': len(cases),
            'success_rate': sum(int(r['success']) for r in rows) / len(rows) if complete else None,
            'stop': reason or 'complete', 'wall_s': time.monotonic() - started,
            'training_started': False, **counters}
