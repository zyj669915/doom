# -*- coding: utf-8 -*-
"""走廊管线纯函数单测（报告 §8.5 第 8 条）。

不启动 ViZDoom、不依赖 torch：只测 corridor_common 的纯函数原语。
运行方式（任一）：
  python tests/test_corridor_pipeline.py      # 自带极简 runner
  python -m pytest tests/test_corridor_pipeline.py -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from corridor_common import (  # noqa: E402
    EPS_STD, FeatureWindow, action_F, classification_metrics, fit_stats,
    frames_of_group, hash_or_weak, make_groups, normalize, rankpct, reset_brain,
    run_id, select_topk, select_topk_unsupervised, split_by_group,
    write_provenance,
)

RNG = np.random.RandomState(0)


def _rand_counts(T=120, n=60, seed=1):
    r = np.random.RandomState(seed)
    return r.poisson(2.0, size=(T, n)).astype(np.float32)


# --------------------------------------------------------------------------
# bug B：离线/在线窗口逐元素一致 + 旧绝对 tics 索引的回归
# --------------------------------------------------------------------------
def test_window_online_equals_offline():
    counts = _rand_counts(T=120, n=60, seed=7)
    idx = np.arange(0, 60, 2)
    for win, diff in ((1, False), (5, True), (5, False), (3, True)):
        w = FeatureWindow(idx, win=win, diff=diff)
        F_off = w.build(counts)
        assert F_off.shape == (120, w.dim)
        w.reset()
        F_on = np.stack([w.step(counts[t, idx]) for t in range(120)])
        assert np.array_equal(F_on, F_off), f'win={win} diff={diff} 在线与离线不一致'


def test_window_offline_semantics_match_old_build():
    """旧 build 语义：slot j = 绝对帧 max(0, t-win+1+j)；差分 v[i0]-v[i0-1]。"""
    counts = _rand_counts(T=30, n=8, seed=3)
    idx = np.array([1, 3, 5], dtype=np.int64)
    win, diff = 5, True
    w = FeatureWindow(idx, win=win, diff=diff)
    F = w.build(counts)
    d = len(idx)
    # 手工重算 t=9 的第 0 个 slot（应为帧 5）
    t = 9
    for j in range(win):
        i0 = max(0, t - win + 1 + j)
        base = 0
        v = counts[i0, idx]
        v0 = counts[i0 - 1, idx] if i0 > 0 else np.zeros(d, np.float32)
        assert np.array_equal(F[t, j * d * 2:(j * d * 2) + d], v)
        assert np.array_equal(F[t, j * d * 2 + d:(j + 1) * d * 2], v - v0)
        del base


def test_window_regression_old_absolute_tics_bug():
    """回归：旧代码用绝对 tics 索引短 hist 并夹到 len-1，导致窗口槽位重复
    当前帧。正确实现下，win=5 窗口的第 0 槽永远是 t-4 帧，不是 t 帧。"""
    counts = _rand_counts(T=20, n=10, seed=11)
    idx = np.arange(5, dtype=np.int64)
    w = FeatureWindow(idx, win=5, diff=False)
    F = w.build(counts)
    t = 9
    # 第 0 槽 = 帧 t-4=5
    assert np.array_equal(F[t, :5], counts[5, idx])
    # 最后一个槽 = 当前帧 t
    assert np.array_equal(F[t, 4 * 5:5 * 5], counts[9, idx])


def test_window_first_frame_padding_semantics():
    """首帧填充语义：t=0 时所有槽位取帧 0，差分槽 = 当前值 - 0（旧 build 同语义）。"""
    counts = _rand_counts(T=3, n=8, seed=5)
    idx = np.array([0, 2, 4], dtype=np.int64)
    w = FeatureWindow(idx, win=5, diff=True)
    F = w.build(counts)
    d = len(idx)
    for j in range(5):
        hv = F[0, j * d * 2:(j * d * 2) + d]
        df = F[0, j * d * 2 + d:(j + 1) * d * 2]
        assert np.array_equal(hv, counts[0, idx]), f'slot {j} 应为帧 0'
        assert np.array_equal(df, hv), f'slot {j} 差分应为当前值（首帧填充）'


# --------------------------------------------------------------------------
# bug A：归一化统计只在原始特征上拟合；旧「先 zscore 再存 0/1」路径必须失败
# --------------------------------------------------------------------------
def test_normalize_roundtrip_and_unit_variance():
    F = _rand_counts(T=200, n=32, seed=9).astype(np.float32)
    mu, sd = fit_stats(F)
    Z = normalize(F, mu, sd)
    # 归一化后近似零均值单位方差
    assert abs(Z.mean()) < 0.02
    assert abs(Z.std() - 1.0) < 0.02
    # 往返：用同一组统计可还原
    R = Z * sd + mu
    assert np.allclose(R, F, atol=1e-5)


def test_old_bug_fit_stats_on_already_zscored_fails_to_normalize_raw():
    """旧 bug A 路径：对已 zscore 的特征再 fit_stats → mu≈0/sd≈1，
    用这组统计归一化【原始】特征无法得到单位方差。"""
    F = _rand_counts(T=200, n=32, seed=13).astype(np.float32)
    mu_good, sd_good = fit_stats(F)                       # 正确路径
    Z_old = normalize(F, mu_good, sd_good)                # 训练时特征
    mu_bug, sd_bug = fit_stats(Z_old)                     # 旧代码存的是这组
    assert abs(mu_bug.mean()) < 0.02 and abs(sd_bug.mean() - 1.0) < 0.02
    # 旧路径闭环：用 bug 统计归一化原始特征 → 方差远离 1
    Z_bug_closed = normalize(F, mu_bug, sd_bug)
    assert abs(Z_bug_closed.std() - 1.0) > 0.05
    # 训练特征与闭环特征不一致（这正是「离线 0.565、闭环不动」的根因）
    assert not np.allclose(Z_old[:50], Z_bug_closed[:50])


# --------------------------------------------------------------------------
# bug C/E/F：分组与无泄漏划分
# --------------------------------------------------------------------------
def test_make_groups_shape_and_order():
    g = make_groups(1000, [41027, 41028, 41029, 41030], 250)
    assert g.shape == (1000,)
    assert g[0] == 41027 and g[249] == 41027 and g[250] == 41028 and g[999] == 41030
    try:
        make_groups(999, [10, 11], 500)
        assert False, '应拒绝帧数与组数不匹配'
    except ValueError:
        pass


def test_split_by_group_no_overlap_and_exhaustive():
    g = make_groups(2000, [41027, 41028, 41029, 41030], 500)
    tr, va, te = split_by_group(g, [41027, 41028], [41030], val_ids=[41029])
    assert tr.sum() == 1000 and va.sum() == 500 and te.sum() == 500
    assert tr.sum() + va.sum() + te.sum() == 2000
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    # 重叠组必须报错
    try:
        split_by_group(g, [41027], [41027])
        assert False, '重叠组应报错'
    except ValueError:
        pass


def test_frames_of_group():
    g = make_groups(1000, [41027, 41028], 500)
    m = frames_of_group(g, [41027])
    assert m.sum() == 500 and m[:500].all() and not m[500:].any()


def test_selection_fit_on_train_subset_only():
    """选择函数只接收训练子集：验证 select_topk 对 (训练子集) 的调用路径
    不接触测试帧——直接用掩码取子集后调用，索引可复用。"""
    counts = _rand_counts(T=2000, n=100, seed=21)
    acts = RNG.randint(0, 3, size=2000).astype(np.int64)
    g = make_groups(2000, [0, 1, 2, 3], 500)
    tr = frames_of_group(g, [0, 1, 2])
    idx = select_topk(counts[tr], acts[tr], k=16, ncls=3)
    assert idx.dtype == np.int64 and idx.shape == (16,)
    assert np.all(idx >= 0) and np.all(idx < 100)
    # 无监督版同样只吃训练子集
    idx2 = select_topk_unsupervised(counts[tr], k=16)
    assert idx2.shape == (16,)


# --------------------------------------------------------------------------
# bug H：指标口径
# --------------------------------------------------------------------------
def test_classification_metrics_consistent():
    Y = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)  # 类 0 占 40%
    pred = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    m = classification_metrics(Y, pred, ncls=3)
    assert m['n_observed_classes'] == 3
    # 指标已 round 到 4 位，比较用 1e-4 容差
    assert abs(m['balanced_chance'] - 1 / 3) < 1e-4
    # 逐类 recall：类0=3/4, 类1=3/3, 类2=3/3 → balanced=mean
    exp = (3 / 4 + 3 / 3 + 3 / 3) / 3
    assert abs(m['balanced_acc'] - exp) < 1e-4
    assert abs(m['ordinary_acc'] - 0.9) < 1e-4
    assert abs(m['majority_frac'] - 0.4) < 1e-4
    assert len(m['confusion']) == 3 and all(len(r) == 3 for r in m['confusion'])


def test_classification_metrics_empty_class():
    """bug H：场景声明 6 类但只有 3 类出现时，balanced 对 1/C 而非 1/6。"""
    Y = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    pred = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    m = classification_metrics(Y, pred, ncls=6)
    assert m['n_observed_classes'] == 3
    assert abs(m['balanced_chance'] - 1 / 3) < 1e-4
    assert m['per_class_recall'][3] is None
    assert m['per_class_recall'][4] is None
    assert m['per_class_recall'][5] is None


# --------------------------------------------------------------------------
# bug J：episode 边界（FQI 转移不跨组、窗口不跨组）
# --------------------------------------------------------------------------
def test_window_built_per_group_does_not_cross_boundary():
    """把 4 段数据拼成一条再整段 build（旧行为）会在边界处混入前一段帧；
    按组分别 build 再拼接（修复后）不会。"""
    n, d = 40, 6
    idx = np.arange(d, dtype=np.int64)
    counts_all = np.zeros((4 * n, 12), np.float32)
    for s in range(4):
        counts_all[s * n:(s + 1) * n] = _rand_counts(T=n, n=12, seed=100 + s)
    # 修复前：整段 build（窗口跨越组边界）
    w_bad = FeatureWindow(idx, win=5, diff=True)
    F_bad = w_bad.build(counts_all)
    # 修复后：按组 build 再拼接
    w_good = FeatureWindow(idx, win=5, diff=True)
    parts = [w_good.build(counts_all[s * n:(s + 1) * n]) for s in range(4)]
    F_good = np.concatenate(parts, axis=0)
    # 组内、窗口不触碰边界（t ≥ 组起点+win-1）时两法一致
    assert np.array_equal(F_bad[n + 5], F_good[n + 5])
    # 边界帧（每组第 0 帧）不同：bad 窗口混入上一组末尾帧
    assert not np.array_equal(F_bad[n], F_good[n])


def test_fqi_boundary_mask():
    """FQI 的组内转移掩码：每组最后一帧视为 terminal，s' 不跨组。"""
    groups = np.repeat(np.arange(4, dtype=np.int64), 50)
    is_last = np.zeros(len(groups), dtype=bool)
    is_last[:-1] = groups[:-1] != groups[1:]
    is_last[-1] = True
    assert is_last[49] and is_last[99] and is_last[149] and is_last[199]
    assert not is_last[48] and not is_last[50]


# --------------------------------------------------------------------------
# bug K：脑状态重置
# --------------------------------------------------------------------------
def test_reset_brain_restores_initial_state():
    n = 100
    brain = type('FakeBrain', (), {})()
    brain.n = n
    brain.retina = np.arange(10, dtype=np.int32)
    brain.lamina = np.arange(10, 20, dtype=np.int32)
    brain.sugar = np.array([], dtype=np.int32)
    brain.cursor = 500
    brain.v = np.random.rand(n).astype(np.float32) * 40 - 60
    brain.g = np.random.rand(n).astype(np.float32)
    brain.drive = np.ones(n, np.float32)
    brain.refractory = np.random.randint(0, 10, n).astype(np.int16)
    brain.queue = np.random.randint(0, n, (20, n)).astype(np.int32)
    brain.queue_count = np.random.randint(0, 5, 20).astype(np.int32)
    brain.counts = np.random.randint(0, 100, n).astype(np.int32)
    brain.luminance = np.random.rand(10).astype(np.float32)
    brain.active = np.random.randint(0, n, n).astype(np.int32)
    brain.active_flag = np.ones(n, np.uint8)
    brain.nactive = np.array([n], dtype=np.int32)
    brain.total_spikes = 12345
    brain.sim_ms = 99.0

    reset_brain(brain)

    assert brain.cursor == 0
    assert np.all(brain.v == -52.0)
    assert np.all(brain.g == 0.0)
    assert np.all(brain.drive == 0.0)
    assert np.all(brain.refractory == 0)
    assert np.all(brain.queue == 0) and np.all(brain.queue_count == 0)
    assert np.all(brain.counts == 0)
    assert np.all(brain.luminance == 0.0)
    assert brain.nactive[0] == 20  # retina(10) ∪ lamina(10)
    assert brain.total_spikes == 0 and brain.sim_ms == 0.0


# --------------------------------------------------------------------------
# bug L：provenance / run_id
# --------------------------------------------------------------------------
def test_run_id_unique_and_format():
    a, b = run_id('stu'), run_id('stu')
    assert a != b
    # 形如 stu-20261018-235552-42f9248c03（前缀-日期-时间-随机段）
    assert a.startswith('stu-') and len(a.split('-')) == 4


def test_provenance_writes_json_with_hashes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / 'a.py'
        src.write_text('print(1)\n', encoding='utf-8')
        out = tmp / 'run1'
        rec = write_provenance(out, {'run_id': 'r1', 'seed': 7},
                               sources={'code': src, 'missing': tmp / 'nope.py'},
                               extra={'n': 3})
        assert rec['run_id'] == 'r1'
        assert rec['source_hashes']['code']['sha256'] == hash_or_weak(src)['sha256']
        assert rec['source_hashes']['missing'] is None
        assert rec['extra']['n'] == 3
        loaded = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
        assert loaded['config']['seed'] == 7


# --------------------------------------------------------------------------
# 简单 runner（无 pytest 时）
# --------------------------------------------------------------------------
def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL {fn.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    _main()
