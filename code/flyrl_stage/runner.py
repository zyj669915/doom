"""独立三arm闭环、严格来源、限额训练与冻结评估。"""
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from flyrl.environment import GameEnv
from flyrl.runner import sha256, json_write, GRAPH_SHA256
from flyrl_next.learning import DQNAgent
from .config import ROOT, load_config, Curriculum
from .encoding import Pipeline
from .resources import gate, MemoryMonitor


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def provenance(c):
    sources = {}
    for folder in ('experiment/code/flyrl_stage', 'experiment/code/flyrl_next',
                   'experiment/code/flyrl', 'doomfly/doom'):
        for p in (ROOT / folder).rglob('*.py'):
            sources[str(p.relative_to(ROOT))] = sha256(p)
    for name in ('flyrl_stage_cli.py', 'corridor_common.py'):
        p = ROOT / 'experiment/code' / name
        sources[str(p.relative_to(ROOT))] = sha256(p)
    import vizdoom
    inputs = {'iwad': Path(vizdoom.__file__).parent / 'freedoom2.wad',
              **{k: resolve(c[k]) for k in ('graph', 'neurons', 'catalog', 'annotations')}}
    result = {'sources': sources,
              'inputs': {k: {'path': str(v), 'sha256': sha256(v)} for k, v in inputs.items()},
              'versions': {k: importlib.metadata.version(k) for k in
                           ('numpy', 'torch', 'numba', 'vizdoom', 'pandas', 'pyarrow')}}
    if result['inputs']['graph']['sha256'] != GRAPH_SHA256:
        raise ValueError('权威图哈希不匹配')
    return result


def new_run(c, kind):
    path = resolve(c['output']) / (time.strftime('%Y%m%d-%H%M%S') +
                                 f'-{kind}-{c["arm"]}-' + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    json_write(path / 'config.json', c)
    return path


def append(path, row):
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def actions(c):
    return list(range(6))


def make_env(c, path, layout, seed):
    return GameEnv('corridor', seed, c['max_tics'], layout, max_turn=c['max_turn'],
                   gamma=c['gamma_tic'], rewards=c['rewards'], run_dir=path)


def macro_step(env, pipeline, action, c, remaining=None):
    """逐tic推进；truncation保留bootstrap，不将遥测传入策略。"""
    limit = min(c['action_repeat'], remaining) if remaining is not None else c['action_repeat']
    if limit < 1:
        raise ValueError('宏步长度必须为正')
    reward = raw_return = brain_ms = 0.
    terms = {}
    for i in range(limit):
        rgb, r, term, trunc, info = env.step(action)
        reward += c['gamma_tic'] ** i * r
        raw_return += r
        for key, value in info['reward_terms'].items():
            terms[key] = terms.get(key, 0.) + value
        if rgb is not None:
            pipeline.tic(rgb)
            brain_ms += float(pipeline.last_diagnostic.get('actual_ms', 0.))
        if term or trunc:
            break
    obs = pipeline.observation()
    if not np.isfinite(reward) or not all(np.isfinite(v).all() for v in obs.values()):
        raise FloatingPointError('奖励或策略特征出现非有限值')
    k = i + 1
    info.update(macro_tics=k, raw_reward=raw_return, macro_reward_terms=terms, brain_ms=brain_ms)
    return obs, reward, 0. if term else c['gamma_tic'] ** k, term, trunc, info


def make_agent(c, pipeline):
    return DQNAgent({**pipeline.spec, 'actions': 6},
                    config={**c['learner'], 'device': 'cpu', 'seed': c['seed']})


def checkpoint(path, c, prov, pipeline, agent, state):
    name = f'checkpoint-{state["game_tics"]:09d}-{uuid.uuid4().hex[:8]}'
    temp = path / (name + '.incomplete')
    temp.mkdir()
    torch.save({'pipeline': pipeline.state_dict(), 'learner': agent.state_dict(),
                'state': state, 'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state()},
               temp / 'state.pt')
    json_write(temp / 'manifest.json', {'config': c, 'provenance': prov,
               'sha256': sha256(temp / 'state.pt'),
               'resume_semantics': 'new_episode; brain/game/history reset; not exact game restoration'})
    final = path / name
    os.replace(temp, final)
    json_write(path / 'latest-checkpoint.json', {'checkpoint': str(final)})
    return final


def load_checkpoint(path, c=None):
    p = Path(path)
    manifest = json.loads((p / 'manifest.json').read_text(encoding='utf-8'))
    saved = load_config(overrides=manifest['config'])
    if c is not None:
        ignored = {'wall_hours', 'max_game_tics', 'output'}
        if {k: v for k, v in c.items() if k not in ignored} != {k: v for k, v in saved.items() if k not in ignored}:
            raise ValueError('续训配置与冻结协议不匹配')
    if sha256(p / 'state.pt') != manifest['sha256'] or provenance(saved) != manifest['provenance']:
        raise ValueError('checkpoint哈希、源码、版本或输入漂移；拒绝加载')
    data = torch.load(p / 'state.pt', map_location='cpu', weights_only=True)
    return saved, manifest, data


def stop_reason(c, path, monitor, elapsed):
    if monitor.stop_requested:
        return 'resource_stop'
    if any((p / 'STOP').exists() for p in (path, *list(path.parents)[:2])):
        return 'controlled_stop'
    if elapsed >= c['wall_hours'] * 3600:
        return 'wall_limit'
    return None


def guarded(c, kind, operation):
    """API入口也检查门禁；CLI另在导入大库前检查一次。"""
    c = load_config(overrides=c)
    path = new_run(c, kind)
    admission = gate(c['ram_gib'], c['commit_gib'])
    json_write(path / 'admission.json', admission)
    if not admission['allowed']:
        result = {'run': str(path), 'stop': 'resource_gate_rejected', 'started': False,
                  'errors': admission['errors']}
        json_write(path / 'result.json', result)
        return result
    latest = None
    with MemoryMonitor(path / 'resources.json') as monitor:
        try:
            if monitor.stop_requested:
                raise RuntimeError('初始化前资源监控请求停止')
            result = operation(c, path, monitor)
            json_write(path / 'result.json', result)
            return result
        except BaseException as exc:
            pointer = path / 'latest-checkpoint.json'
            if pointer.exists():
                latest = json.loads(pointer.read_text(encoding='utf-8'))['checkpoint']
            json_write(path / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                                              'last_checkpoint': latest})
            raise


def train(c, resume=None, smoke=False):
    return guarded(c, 'accept' if smoke else 'train',
                   lambda c, path, monitor: _train(c, path, monitor, resume, smoke))


def _train(c, path, monitor, resume, smoke):
    random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = time.monotonic()
    prov = provenance(c)
    json_write(path / 'provenance.json', prov)
    # 来源校验先于载入checkpoint及完整脑。
    data = load_checkpoint(resume, c)[2] if resume else None
    if stop_reason(c, path, monitor, time.monotonic() - started):
        raise RuntimeError('来源核对后资源/墙钟/STOP要求停止；不载入脑')
    pipeline = Pipeline(c, path)
    agent = make_agent(c, pipeline)
    curriculum = Curriculum(c)
    rng = np.random.default_rng(c['seed'])
    state = {'game_tics': 0, 'decisions': 0, 'episodes': 0, 'wall_s': 0.,
             'calibration_tics': 0, 'brain_ms': 0., 'smoke_only': smoke,
             'initial_observation_tics': 0}
    if data is not None:
        pipeline.load_state_dict(data['pipeline'])
        agent.load_state_dict(data['learner'])
        state = data['state']
        if state['smoke_only'] or not pipeline.calibrated:
            raise ValueError('无校准/优化验收快照不能作为正式续训起点')
        rng.bit_generator.state = state['numpy_rng']
        curriculum.load_state_dict(state['curriculum'])
        random.setstate(data['python_rng'])
        torch.set_rng_state(data['torch_rng'])
        del data
    previous_wall = state['wall_s']
    env = None
    ep = None
    reason = None
    latest = None

    def stopping():
        return stop_reason(c, path, monitor, previous_wall + time.monotonic() - started)

    def save():
        state.update(curriculum=curriculum.state_dict(), numpy_rng=rng.bit_generator.state,
                     wall_s=previous_wall + time.monotonic() - started, updates=agent.updates,
                     replay_size=len(agent.replay))
        saved = checkpoint(path, c, prov, pipeline, agent, state)
        json_write(path / 'status.json', {'checkpoint': str(saved), **state})
        return saved

    try:
        if not resume and not smoke:
            for index, layout in enumerate(c['train_layouts']):
                quota = c['calibration_steps'] // len(c['train_layouts']) + int(index < c['calibration_steps'] % len(c['train_layouts']))
                if env is not None:
                    env.close()
                env = None
                for _ in range(quota):
                    if stopping():
                        raise RuntimeError('校准未完成即触发安全/墙钟停止；不可用于正式续训')
                    if env is None or env._done:
                        if env is not None:
                            env.close()
                        env = make_env(c, path, layout, c['train_seeds'][index % len(c['train_seeds'])])
                        rgb, _ = env.reset()
                        pipeline.reset()
                        pipeline.tic(rgb)
                        state['initial_observation_tics'] += 1
                    *_, info = macro_step(env, pipeline, int(rng.integers(6)), c)
                    pipeline.calibration_update()
                    state['calibration_tics'] += info['macro_tics']
            pipeline.freeze_calibration()
            json_write(path / 'calibration.json', {'samples': pipeline.calibration_count,
                'game_tics': state['calibration_tics'], 'dimension': 1536,
                'constant_dimensions': int(np.count_nonzero(pipeline.sd <= .0100001)),
                'scope': 'train-only; all train layouts; random actions'})
            env.close()
            env = None
        # 0点只存快照，不在训练进程中再分配第二个脑。
        if not smoke and not resume:
            latest = save()
            append(path / 'evaluation-queue.jsonl', {'game_tics': 0, 'checkpoint': str(latest), 'quick': True})
        while state['game_tics'] < c['max_game_tics']:
            reason = stopping()
            if reason:
                break
            # 资源验收总推进不超过256tic，包含初始感觉帧。
            if smoke and state['game_tics'] + state['initial_observation_tics'] >= 256:
                reason = 'accept_tic_limit'
                break
            if env is None or env._done:
                if env is not None:
                    env.close()
                layout = curriculum.select(rng)
                env = make_env(c, path, layout, c['train_seeds'][state['episodes'] % len(c['train_seeds'])])
                rgb, _ = env.reset()
                pipeline.reset()
                pipeline.tic(rgb)
                state['initial_observation_tics'] += 1
                obs = pipeline.observation()
                ep = {'episode': state['episodes'], 'return': 0., 'tics': 0,
                      'max_progress': 0., 'action_counts': [0] * 6, 'layout': layout}
            epsilon = c['epsilon_end'] + (1 - c['epsilon_end']) * max(0., 1 - state['game_tics'] / c['epsilon_decay_steps'])
            action = agent.act(obs, epsilon=epsilon)
            remaining = c['max_game_tics'] - state['game_tics']
            next_point = next((p for p in c['evaluation_points'] if p > state['game_tics']), c['max_game_tics'])
            remaining = min(remaining, next_point - state['game_tics'])
            if smoke:
                remaining = min(remaining, 256 - state['game_tics'] - state['initial_observation_tics'])
                if remaining <= 0:
                    reason = 'accept_tic_limit'
                    break
            nxt, reward, discount, term, trunc, info = macro_step(env, pipeline, action, c, remaining)
            k = info['macro_tics']
            ep['return'] += info['raw_reward']
            ep['tics'] += k
            ep['action_counts'][action] += k
            ep['max_progress'] = max(ep['max_progress'], info.get('progress') or 0.)
            state['game_tics'] += k
            state['decisions'] += 1
            state['brain_ms'] += info['brain_ms']
            curriculum.advance(k)
            if not smoke:
                agent.observe(obs, action, reward, nxt, discount, terminated=term, truncated=trunc)
                metrics = agent.update() if not stopping() else None
                if metrics is not None:
                    if any(not np.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError('学习器指标非有限')
                    append(path / 'updates.jsonl', {'game_tics': state['game_tics'], **metrics})
            obs = nxt
            if state['decisions'] % 100 == 0:
                append(path / 'health.jsonl', {'game_tics': state['game_tics'],
                    'epsilon': epsilon, 'vector_clip_fraction': float(np.mean(np.abs(obs['vector']) >= 10)),
                    'vector_std': float(np.std(obs['vector'])), **pipeline.last_diagnostic})
            if term or trunc:
                curriculum.record(layout, info['success'])
                append(path / 'episodes.jsonl', {**ep, 'success': info['success'],
                    'terminated': term, 'truncated': trunc, 'wall_hits': info['wall_hits']})
                state['episodes'] += 1
            point = state['game_tics'] in c['evaluation_points']
            if point or state['decisions'] % c['checkpoint_every'] == 0:
                latest = save()
                if point and not smoke:
                    append(path / 'evaluation-queue.jsonl', {'game_tics': state['game_tics'],
                           'checkpoint': str(latest), 'quick': state['game_tics'] < 50000})
        latest = save()
        return {'run': str(path), 'checkpoint': str(latest), 'stop': reason or 'game_tic_limit',
                **state, 'started': True, 'optimized': not smoke and agent.updates > 0,
                'partial_episode': ep, 'resume_semantics': 'new_episode'}
    finally:
        if env is not None:
            env.close()


def task_probe(c):
    """固定/周期动作负对照；动作不读取RGB或遥测，不训练、不加载脑。"""
    return guarded(c, 'task-probe', _task_probe)


def _task_probe(c, path, monitor):
    started = time.monotonic()
    policies = [(f'fixed-{a}', lambda t, a=a: a) for a in range(6)]
    policies += [('periodic-left', lambda t: 3 if t % 400 < 300 else 1),
                 ('periodic-right', lambda t: 3 if t % 400 < 300 else 5)]
    scores = {}
    for name, policy in policies:
        successes = []
        for index, layout in enumerate(c['train_layouts']):
            env = make_env(c, path, layout, c['train_seeds'][index % len(c['train_seeds'])])
            try:
                env.reset()
                for tic in range(c['max_tics']):
                    if stop_reason(c, path, monitor, time.monotonic() - started):
                        return {'run': str(path), 'stop': 'controlled_or_resource_or_wall_stop',
                                'complete': False, 'scores': scores}
                    _, _, term, trunc, info = env.step(policy(tic))
                    if term or trunc:
                        successes.append(bool(info['success']))
                        append(path / 'episodes.jsonl', {'policy': name, 'layout': layout,
                            'success': info['success'], 'tics': tic + 1, 'wall_hits': info['wall_hits']})
                        break
            finally:
                env.close()
        scores[name] = float(np.mean(successes))
    trivial = max(scores.values()) >= .75
    return {'run': str(path), 'stop': 'task_too_easy' if trivial else 'complete',
            'complete': True, 'acceptable_for_pilot': not trivial, 'scores': scores,
            'scope': 'train layouts only; eight predeclared open-loop policies; not exhaustive controllability proof'}


def evaluate(checkpoint_path, split='val', quick=False, policy='learned'):
    if split not in ('val', 'test') or policy not in ('learned', 'untrained', 'random'):
        raise ValueError('未知评估模式')
    # 先读JSON取得资源预算；大state文件在门禁后加载。
    manifest = json.loads((Path(checkpoint_path) / 'manifest.json').read_text(encoding='utf-8'))
    c = load_config(overrides=manifest['config'])
    return guarded(c, 'eval-' + split, lambda c, path, monitor:
                   _evaluate(c, path, monitor, checkpoint_path, split, quick, policy))


def _evaluate(c, path, monitor, checkpoint_path, split, quick, policy):
    _, manifest, data = load_checkpoint(checkpoint_path)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    pipeline = Pipeline(c, path)
    pipeline.load_state_dict(data['pipeline'])
    agent = make_agent(c, pipeline)
    if policy != 'untrained':
        agent.load_state_dict(data['learner'], restore_optimizer=False, restore_replay=False)
    training_tics = data['state']['game_tics']
    del data
    if not pipeline.calibrated:
        raise ValueError('冻结评估要求训练校准完成')
    rng = np.random.default_rng(c['seed'] + 999)
    layouts = c[split + '_layouts'][:4] if quick else c[split + '_layouts']
    episodes = []
    started = time.monotonic()
    reason = None
    for index, layout in enumerate(layouts):
        if stop_reason(c, path, monitor, time.monotonic() - started):
            reason = 'controlled_or_resource_or_wall_stop'
            break
        env = make_env(c, path, layout, c[split + '_seeds'][index % len(c[split + '_seeds'])])
        try:
            rgb, _ = env.reset()
            pipeline.reset()
            pipeline.tic(rgb)
            obs = pipeline.observation()
            row = {'layout': layout, 'seed': c[split + '_seeds'][index % len(c[split + '_seeds'])],
                   'return': 0., 'tics': 0, 'max_progress': 0., 'action_counts': [0] * 6}
            while True:
                if stop_reason(c, path, monitor, time.monotonic() - started):
                    reason = 'controlled_or_resource_or_wall_stop'
                    break
                action = int(rng.integers(6)) if policy == 'random' else agent.act(obs, epsilon=0)
                obs, _, _, term, trunc, info = macro_step(env, pipeline, action, c)
                row['tics'] += info['macro_tics']
                row['return'] += info['raw_reward']
                row['action_counts'][action] += info['macro_tics']
                row['max_progress'] = max(row['max_progress'], info.get('progress') or 0.)
                if term or trunc:
                    row.update(success=info['success'], wall_hits=info['wall_hits'])
                    episodes.append(row)
                    append(path / 'episodes.jsonl', row)
                    break
            if reason:
                json_write(path / 'partial-episode.json', row)
                break
        finally:
            env.close()
    complete = len(episodes) == len(layouts)
    return {'run': str(path), 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': manifest['sha256'],
            'split': split, 'quick': quick, 'policy': policy, 'episodes': episodes,
            'training_game_tics': training_tics, 'complete': complete, 'frozen': True,
            'stop': reason or 'complete',
            'success_rate': float(np.mean([x['success'] for x in episodes])) if complete else None}
