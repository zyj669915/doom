"""逐 tic 闭环、显式半马尔可夫折扣和安全新局续训。"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid

import numpy as np
import torch

from flyrl.environment import GameEnv
from flyrl.runner import sha256, signature, json_write, GRAPH_SHA256
from .config import ROOT, load_config, Curriculum
from .encoding import Pipeline
from .learning import DQNAgent, PPOAgent


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def provenance(c):
    sources = {}
    for folder in ('experiment/code/flyrl_next', 'experiment/code/flyrl', 'doomfly/doom'):
        for p in (ROOT / folder).rglob('*.py'):
            sources[str(p.relative_to(ROOT))] = sha256(p)
    for name in ('flyrl_next_cli.py', 'corridor_common.py'):
        p = ROOT / 'experiment/code' / name
        sources[str(p.relative_to(ROOT))] = sha256(p)
    import vizdoom
    inputs = {'iwad': Path(vizdoom.__file__).parent / 'freedoom2.wad'}
    if c['route'] in ('A1', 'A2', 'A3', 'A7'):
        inputs.update({k: resolve(c[k]) for k in ('graph', 'neurons', 'catalog', 'annotations')})
    if c['task'] == 'combat':
        inputs['combat_wad'] = ROOT / 'doomfly/doom/scenarios/combat_survival.wad'
    result = {'sources': sources, 'inputs': {k: {'path': str(v), 'sha256': sha256(v)} for k, v in inputs.items()},
              'versions': {k: importlib.metadata.version(k) for k in ('numpy', 'torch', 'numba', 'vizdoom', 'pandas', 'pyarrow')}}
    if 'graph' in result['inputs'] and result['inputs']['graph']['sha256'] != GRAPH_SHA256:
        raise ValueError('权威图哈希不匹配')
    return result


def new_run(c, kind):
    p = resolve(c['output']) / (time.strftime('%Y%m%d-%H%M%S') + f'-{kind}-{c["route"]}-' + uuid.uuid4().hex[:8])
    p.mkdir(parents=True, exist_ok=False)
    json_write(p / 'config.json', c)
    return p


def append(path, row):
    with open(path, 'a', encoding='utf-8') as h:
        h.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def actions(c):
    full = 6 if c['task'] == 'corridor' else 12
    from flyrl.environment import ActionCodec
    codec = ActionCodec(c['task'], c['max_turn'])
    return [i for i in range(full) if c['action_set'] == 'full' or codec.decode(i)['forward'] > 0]


def make_env(c, path, layout, seed):
    return GameEnv(c['task'], seed, c['max_tics'], layout if c['task'] == 'corridor' else None,
                   max_turn=c['max_turn'], gamma=c['gamma_tic'], rewards=c['rewards'], run_dir=path)


def macro_step(env, pipeline, action, c, remaining=None, intervention='none'):
    """不跳过中间感觉帧；返回完整 bootstrap discount，而非额外 mask。"""
    limit = min(c['action_repeat'], remaining) if remaining is not None else c['action_repeat']
    reward = raw_return = 0.0
    terms = {}
    brain_ms = 0.0
    previous_progress = env._info(env._telemetry).get('progress')
    for i in range(limit):
        rgb, r, terminated, truncated, info = env.step(action)
        if c['objective'] == 'progress_delta' and c['task'] == 'corridor':
            delta = 5.0 * (info['progress'] - previous_progress)
            r += delta - info['reward_terms'].get('potential', 0.0)
            info['reward_terms'] = {k: v for k, v in info['reward_terms'].items() if k != 'potential'}
            info['reward_terms']['progress_delta'] = delta
        previous_progress = info.get('progress')
        reward += c['gamma_tic'] ** i * r
        raw_return += r
        for key, value in info['reward_terms'].items():
            terms[key] = terms.get(key, 0.0) + value
        if rgb is not None:
            pipeline.tic(rgb)
            brain_ms += float(pipeline.last_diagnostic.get('actual_ms', 0.))
        if terminated or truncated:
            break
    k = i + 1
    obs = pipeline.observation()
    discount = 0.0 if terminated else c['gamma_tic'] ** k
    info.update(macro_tics=k, raw_reward=raw_return, macro_reward_terms=terms, brain_ms=brain_ms)
    return obs, reward, discount, terminated, truncated, info


def make_agent(c, pipeline):
    spec = {**pipeline.spec, 'actions': len(actions(c))}
    cls = DQNAgent if c['algorithm'] == 'dqn' else PPOAgent
    defaults = {'replay_capacity': 2000 if pipeline.spec.get('image_shape') else 5000} if c['algorithm'] == 'dqn' else {}
    return cls(spec, config={**defaults, **c['learner'], 'device': c['device'], 'seed': c['seed']})


def checkpoint(path, c, prov, pipeline, agent, state):
    """目录提交为原子边界；未完成目录不会成为 latest。"""
    name = f'checkpoint-{state["game_tics"]:09d}-{uuid.uuid4().hex[:8]}'
    temp = path / (name + '.incomplete')
    temp.mkdir()
    payload = {'pipeline': pipeline.state_dict(), 'learner': agent.state_dict(), 'state': state,
               'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state()}
    if torch.cuda.is_available():
        payload['cuda_rng'] = torch.cuda.get_rng_state_all()
    torch.save(payload, temp / 'state.pt')
    manifest = {'config': c, 'provenance': prov, 'sha256': sha256(temp / 'state.pt'),
                'resume_semantics': 'new_episode; brain/game/history reset; not exact game restoration'}
    json_write(temp / 'manifest.json', manifest)
    final = path / name
    os.replace(temp, final)
    json_write(path / 'latest-checkpoint.json', {'checkpoint': str(final)})
    return final


def load_checkpoint(path, c=None):
    p = Path(path)
    manifest = json.loads((p / 'manifest.json').read_text(encoding='utf-8'))
    saved = load_config(overrides=manifest['config'])
    if c is not None:
        # 扩展预算不改变模型/时间/任务协议。
        ignored = {'wall_hours', 'max_game_tics', 'output'}
        if {k: v for k, v in c.items() if k not in ignored} != {k: v for k, v in saved.items() if k not in ignored}:
            raise ValueError('续训配置与冻结协议不匹配')
    if sha256(p / 'state.pt') != manifest['sha256']:
        raise ValueError('checkpoint 文件哈希不匹配')
    if provenance(saved) != manifest['provenance']:
        raise ValueError('源码、版本或输入漂移；拒绝加载')
    data = torch.load(p / 'state.pt', map_location='cpu', weights_only=True)
    return saved, manifest, data


def _rollout():
    return {k: [] for k in ('obs', 'actions', 'rewards', 'discounts', 'old_logp', 'values', 'next_values', 'episode_end')}


def train(c, resume=None, smoke=False):
    random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    path = new_run(c, 'smoke' if smoke else 'train')
    prov = provenance(c)
    json_write(path / 'provenance.json', prov)
    pipeline = Pipeline(c, path)
    agent = make_agent(c, pipeline)
    curriculum = Curriculum(c)
    rng = np.random.default_rng(c['seed'])
    state = {'game_tics': 0, 'decisions': 0, 'episodes': 0, 'wall_s': 0.0,
             'calibration_tics': 0, 'ppo_discarded_tail': 0, 'brain_ms': 0.0, 'smoke_only': smoke,
             'initial_observation_brain_ms': 0.0}
    if resume:
        _, _, data = load_checkpoint(resume, c)
        pipeline.load_state_dict(data['pipeline'])
        agent.load_state_dict(data['learner'])
        state = data['state']
        if state.get('smoke_only'):
            raise ValueError('smoke 无校准/优化快照不可作为正式续训起点')
        rng.bit_generator.state = state['numpy_rng']
        curriculum.load_state_dict(state['curriculum'])
        random.setstate(data['python_rng'])
        torch.set_rng_state(data['torch_rng'])
        if 'cuda_rng' in data and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(data['cuda_rng'])
    started = time.monotonic()
    previous_wall = state['wall_s']
    env = None
    rollout = _rollout()
    stop_reason = 'game_tic_limit'
    latest = None
    try:
        # 仅 train 布局、随机策略校准；不更新学习器，不缓存全脑 raster。
        if not resume and not smoke and pipeline.spec.get('vector_dim', 0):
            layouts = c['train_layouts'] if c['task'] == 'corridor' else [None]
            for layout_index, layout in enumerate(layouts):
                quota = c['calibration_steps'] // len(layouts) + int(layout_index < c['calibration_steps'] % len(layouts))
                if env is not None:
                    env.close()
                env = None
                for index in range(quota):
                    if env is None or env._done:
                        if env is not None:
                            env.close()
                        env = make_env(c, path, layout, c['train_seeds'][layout_index % len(c['train_seeds'])])
                        rgb, _ = env.reset()
                        pipeline.reset()
                        pipeline.tic(rgb)
                        pipeline.observation()
                    _, _, _, _, _, calibration_info = macro_step(env, pipeline, int(rng.choice(actions(c))), c)
                    pipeline.calibration_update()
                    state['calibration_tics'] += calibration_info['macro_tics']
                    if time.monotonic() - started >= c['wall_hours'] * 3600:
                        raise RuntimeError('开发/训练预算在校准阶段用尽，未开始优化')
            pipeline.freeze_calibration()
            json_write(path / 'calibration.json', {'samples': pipeline.calibration_count,
                       'game_tics': state['calibration_tics'], 'constant_dimensions': int(np.count_nonzero(pipeline.sd <= .0100001)),
                       'dimension': pipeline.spec.get('vector_dim', 0), 'scope': 'train-only; decision-matched random actions; all train layouts'})
            env.close()
            env = None
        while state['game_tics'] < c['max_game_tics']:
            state['wall_s'] = previous_wall + time.monotonic() - started
            if state['wall_s'] >= c['wall_hours'] * 3600 or any((p / 'STOP').exists() for p in (path, *list(path.parents)[:3])):
                stop_reason = 'controlled_stop' if any((p / 'STOP').exists() for p in (path, *list(path.parents)[:3])) else 'wall_limit'
                break
            if env is None or env._done:
                if env is not None:
                    env.close()
                layout = curriculum.select(rng) if c['task'] == 'corridor' else None
                seed = c['train_seeds'][state['episodes'] % len(c['train_seeds'])]
                env = make_env(c, path, layout, seed)
                rgb, info = env.reset()
                pipeline.reset()
                pipeline.tic(rgb)
                state['initial_observation_brain_ms'] += float(pipeline.last_diagnostic.get('actual_ms', 0.))
                obs = pipeline.observation()
                ep = {'episode': state['episodes'], 'return': 0.0, 'discounted_return': 0.0,
                      'tics': 0, 'max_progress': 0.0, 'action_counts': [0] * len(actions(c)), 'reward_terms': {}}
            epsilon = c['epsilon_end'] + (1 - c['epsilon_end']) * max(0., 1 - state['game_tics'] / c['epsilon_decay_steps'])
            if c['algorithm'] == 'dqn':
                action = agent.act(obs, epsilon=epsilon)
            else:
                action, logp, value = agent.act(obs)
            nxt, reward, discount, term, trunc, info = macro_step(env, pipeline, actions(c)[action], c,
                                                                  c['max_game_tics'] - state['game_tics'])
            k = info['macro_tics']
            ep['discounted_return'] += c['gamma_tic'] ** ep['tics'] * reward
            ep['return'] += info['raw_reward']
            ep['tics'] += k
            ep['action_counts'][action] += k
            ep['max_progress'] = max(ep['max_progress'], info.get('progress') or 0.)
            for key, val in info['macro_reward_terms'].items():
                ep['reward_terms'][key] = ep['reward_terms'].get(key, 0.) + val
            state['game_tics'] += k
            state['decisions'] += 1
            state['brain_ms'] += info['brain_ms']
            curriculum.advance(k)
            metrics = None
            if not smoke:
                if c['algorithm'] == 'dqn':
                    agent.observe(obs, action, reward, nxt, discount, terminated=term, truncated=trunc)
                    if previous_wall + time.monotonic() - started < c['wall_hours'] * 3600:
                        metrics = agent.update()
                else:
                    row = (obs, action, reward, discount, logp, value, 0. if term else agent.value(nxt), term or trunc)
                    for key, val in zip(rollout, row):
                        rollout[key].append(val)
                    if len(rollout['actions']) >= c.get('rollout_steps', 256):
                        if previous_wall + time.monotonic() - started < c['wall_hours'] * 3600:
                            metrics = agent.update(rollout)
                            rollout = _rollout()
            if metrics is not None:
                append(path / 'updates.jsonl', {'game_tics': state['game_tics'], **metrics})
            obs = nxt
            if term or trunc:
                curriculum.record(layout, info['success']) if c['task'] == 'corridor' else None
                append(path / 'episodes.jsonl', {**ep, 'success': info['success'], 'terminated': term, 'truncated': trunc,
                                                'wall_hits': info['wall_hits'], 'telemetry': info['telemetry'], 'layout': layout})
                state['episodes'] += 1
            if state['decisions'] % c['checkpoint_every'] == 0:
                # PPO 尾数据不跨快照，明确丢弃并记账，旧 logp 从不重算。
                state['ppo_discarded_tail'] += len(rollout['actions'])
                rollout = _rollout()
                state.update(curriculum=curriculum.state_dict(), numpy_rng=rng.bit_generator.state,
                             wall_s=previous_wall + time.monotonic() - started)
                latest = checkpoint(path, c, prov, pipeline, agent, state)
        state['ppo_discarded_tail'] += len(rollout['actions'])
        state.update(curriculum=curriculum.state_dict(), numpy_rng=rng.bit_generator.state,
                     wall_s=previous_wall + time.monotonic() - started)
        latest = checkpoint(path, c, prov, pipeline, agent, state)
        result = {'run': str(path), 'checkpoint': str(latest), 'stop': stop_reason, **state,
                  'optimized': not smoke, 'resume_semantics': 'new_episode', 'partial_episode': ep if 'ep' in locals() else None}
        json_write(path / 'result.json', result)
        return result
    except BaseException as exc:
        json_write(path / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'last_checkpoint': str(latest) if latest else None})
        raise
    finally:
        if env is not None:
            env.close()


def evaluate(checkpoint_path, split='val', quick=False, intervention='none', policy='learned'):
    c, manifest, data = load_checkpoint(checkpoint_path)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    path = new_run(c, 'eval-' + split)
    pipeline = Pipeline(c, path)
    pipeline.load_state_dict(data['pipeline'])
    if intervention in ('black', 'scramble'):
        pipeline.intervention = intervention
        if pipeline.brain is not None:
            pipeline.vision.vision.intervention = intervention
    agent = make_agent(c, pipeline)
    if policy != 'untrained':
        agent.load_state_dict(data['learner'], restore_optimizer=False, restore_replay=False)
    rng = np.random.default_rng(c['seed'] + 999)
    layouts = c[split + '_layouts'] if c['task'] == 'corridor' else [None]
    if quick:
        layouts = layouts[:4]
    episodes = []
    for index, layout in enumerate(layouts):
        env = make_env(c, path, layout, c[split + '_seeds'][index % len(c[split + '_seeds'])])
        try:
            rgb, info = env.reset()
            pipeline.reset()
            pipeline.tic(rgb)
            obs = pipeline.observation()
            row = {'success': False, 'tics': 0, 'return': 0., 'max_progress': 0., 'action_counts': [0] * len(actions(c)), 'layout': layout,
                   'seed': c[split + '_seeds'][index % len(c[split + '_seeds'])]}
            while True:
                controlled = {k: v.copy() for k, v in obs.items()}
                if intervention in ('brain_zero', 'brain_permute') and 'vector' in controlled:
                    controlled['vector'] = np.zeros_like(controlled['vector']) if intervention == 'brain_zero' else controlled['vector'][::-1].copy()
                if policy == 'random':
                    action = int(rng.integers(len(actions(c))))
                elif c['algorithm'] == 'dqn':
                    action = agent.act(controlled, epsilon=0)
                else:
                    action = agent.act(controlled, deterministic=True)[0]
                obs, _, _, term, trunc, info = macro_step(env, pipeline, actions(c)[action], c, intervention=intervention)
                row['tics'] += info['macro_tics']
                row['return'] += info['raw_reward']
                row['action_counts'][action] += info['macro_tics']
                row['max_progress'] = max(row['max_progress'], info.get('progress') or 0.)
                if term or trunc:
                    break
            row.update(success=info['success'], wall_hits=info['wall_hits'], telemetry=info['telemetry'], assets=info['assets'])
            episodes.append(row)
            append(path / 'episodes.jsonl', row)
        finally:
            env.close()
    result = {'run': str(path), 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': manifest['sha256'],
              'config': c, 'split': split, 'quick': quick, 'intervention': intervention, 'policy': policy,
              'episodes': episodes, 'training_game_tics': data['state']['game_tics'], 'provenance': manifest['provenance'],
              'success_rate': float(np.mean([x['success'] for x in episodes])), 'frozen': True}
    json_write(path / 'result.json', result)
    return result
