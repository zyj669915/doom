"""仅开发限额验收；绝不使用正式小时预算。"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyrl_next.config import load_config
from flyrl_next.runner import train, load_checkpoint, evaluate, ROOT
from flyrl.runner import json_write, sha256


def main():
    results = []
    for route in ('A4', 'A5', 'A6'):
        overrides = {'max_game_tics': 32, 'max_tics': 4, 'wall_hours': 2 / 60,
                     'calibration_steps': 24, 'rollout_steps': 4, 'checkpoint_every': 4}
        if route == 'A6':
            overrides['learner'] = {'batch_size': 4, 'learning_starts': 4, 'replay_capacity': 32, 'target_every': 2}
        else:
            overrides['learner'] = {'epochs': 1, 'minibatch_size': 4}
        c = load_config(ROOT / f'experiment/configs_next/{route}-2h.json', overrides)
        result = train(c)
        _, _, data = load_checkpoint(result['checkpoint'])
        result['updates_verified'] = data['learner']['updates']
        val = evaluate(result['checkpoint'], quick=True)
        result['eval_run'] = val['run']
        # 保持总预算，受控快照恢复新episode；验证状态完整加载而非重新初始化。
        resumed = train(c, resume=result['checkpoint'])
        result['resume_run'] = resumed['run']
        results.append(result)
    historical = json.loads((ROOT / 'experiment/runs/20261019-163749-eval-val-real-bb2261bc07/result.json').read_text(encoding='utf-8'))
    drift = [p for p, expected in historical['sources'].items() if sha256(ROOT / p) != expected]
    index = {'scope': 'development-only; not game competence', 'visual_results': results,
             'v1_source_drift': drift, 'brain_real_game_status': 'blocked: free physical RAM below 3GiB reserve',
             'brain_unit_tests': 'synthetic complete graph only; not real graph execution'}
    json_write(ROOT / 'experiment/runs_next/acceptance-20261019.json', index)
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
