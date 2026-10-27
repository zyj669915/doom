"""只用训练布局的限量开放环诊断，不根据验证集选参数。"""
import time
import numpy as np
from .encoding import Pipeline
from .runner import new_run, make_env, actions
from flyrl.runner import json_write


def diagnose(c, frames=16):
    path = new_run(c, 'diagnose')
    pipe = Pipeline(c, path)
    rng = np.random.default_rng(c['seed'])
    env = make_env(c, path, c['train_layouts'][0] if c['task'] == 'corridor' else None, c['train_seeds'][0])
    rows = []
    started = time.monotonic()
    try:
        rgb, _ = env.reset()
        pipe.reset()
        pipe.tic(rgb)
        rows.append(dict(pipe.last_diagnostic))
        for _ in range(frames):
            rgb, _, term, trunc, _ = env.step(int(rng.choice(actions(c))))
            if rgb is not None:
                pipe.tic(rgb)
                rows.append(dict(pipe.last_diagnostic))
            if term or trunc:
                break
        result = {'run': str(path), 'route': c['route'], 'frontend': c['frontend'],
                  'dynamics': [c['scale'], c['bias'], c['threshold']], 'spec': pipe.spec,
                  'rows': rows, 'wall_s': time.monotonic() - started,
                  'scope': 'train-only open loop; not behavioral or physiological validation'}
        json_write(path / 'result.json', result)
        return result
    finally:
        env.close()
