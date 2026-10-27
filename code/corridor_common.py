# -*- coding: utf-8 -*-
"""走廊管线共享原语（纯 numpy + stdlib，不依赖 torch / vizdoom）。

对应《果蝇MaleCNS连接组闭环游戏实验_分析报告》§5.1 A–L 中可在纯函数层
验证的修复。每个原语都被 tests/test_corridor_pipeline.py 直接测试：

  * FeatureWindow        离线/在线窗口逐元素一致（bug B）
  * fit_stats/normalize  归一化只在训练组拟合、往返可复算（bug A/D）
  * make_groups/split_by_group 按 episode/seed 分组、无泄漏划分（bug C/E/F）
  * select_topk          topK 特征选择只允许在训练组拟合（bug C/G）
  * classification_metrics 指标口径统一（bug H）
  * reset_brain          每 episode 重置脑状态（bug K）
  * run_id / provenance 唯一运行记录 + 源码/WAD/数据哈希（bug L）
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

EPS_STD = 1e-6


# --------------------------------------------------------------------------
# 归一化（bug A / D）：统计只在训练组原始特征上拟合，闭环用同一组统计。
# --------------------------------------------------------------------------
def fit_stats(F):
    """在【原始】特征上拟合 (mu, sd)。调用方必须保证 F 只含训练组数据。"""
    F = np.asarray(F, dtype=np.float32)
    return F.mean(0), F.std(0) + EPS_STD


def normalize(F, mu, sd):
    """用训练组统计归一化【原始】特征。mu/sd 必须来自 fit_stats(原始特征)。"""
    return (np.asarray(F, dtype=np.float32) - mu) / sd


# --------------------------------------------------------------------------
# 分组与无泄漏划分（bug C / E / F）
# --------------------------------------------------------------------------
def make_groups(n_frames, seeds, n_per_seed):
    """把「按 seed 顺序拼接」的帧序列映射到 group id（每 seed 一段）。

    返回 int64 数组，长度 = n_frames；group id 直接取 seed 值本身，
    便于与 split_by_group(train_ids=SEEDS...) 的语义一一对应。
    """
    n_frames = int(n_frames)
    n_per_seed = int(n_per_seed)
    n_seeds = len(seeds)
    if n_frames != n_seeds * n_per_seed:
        raise ValueError(
            f'n_frames={n_frames} != n_seeds({n_seeds}) * n_per_seed({n_per_seed})'
        )
    return np.repeat(np.asarray(seeds, dtype=np.int64), n_per_seed)


def split_by_group(groups, train_ids, test_ids, val_ids=None):
    """按组（episode/seed）划分，返回 (train_mask, val_mask, test_mask)。

    同一 group 的所有帧只会落入一个子集；组 id 重叠会直接报错，
    避免「同一段轨迹同时进训练和验证」的泄漏。
    """
    groups = np.asarray(groups)
    ids = [np.asarray(x, dtype=np.int64) for x in (train_ids, test_ids)]
    if val_ids is not None:
        ids.append(np.asarray(val_ids, dtype=np.int64))
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if len(np.intersect1d(a, b)) > 0:
                raise ValueError('group ids overlap between splits')
    tr = np.isin(groups, ids[0])
    te = np.isin(groups, ids[1])
    va = np.isin(groups, ids[2]) if val_ids is not None else np.zeros_like(tr)
    return tr, va, te


def frames_of_group(groups, group_ids):
    """返回属于 group_ids 的帧的布尔掩码（便于按组取子集）。"""
    return np.isin(np.asarray(groups), np.asarray(group_ids, dtype=np.int64))


# --------------------------------------------------------------------------
# 特征选择（bug C / G）：select_topk 只应在训练组数据上调用。
# --------------------------------------------------------------------------
def rankpct(x):
    x = np.asarray(x)
    return (np.argsort(np.argsort(x)) + 1.0) / len(x)


def action_F(Y, acts, ncls):
    """逐神经元单因素 ANOVA F：组间/组内方差比，越大越能区分动作类。"""
    Y = np.asarray(Y, dtype=np.float32)
    acts = np.asarray(acts, dtype=np.int64)
    n = Y.shape[0]
    onehot = (acts[:, None] == np.arange(ncls)).astype(np.float32)
    gsum = onehot.T @ Y                    # (ncls, d)
    gcnt = onehot.sum(0).clip(min=1e-6)
    Y2 = Y * Y
    gsum2 = onehot.T @ Y2
    ssw = gsum2.sum(0) - (gsum * gsum / gcnt[:, None]).sum(0)
    ssb = (gsum * gsum / gcnt[:, None]).sum(0) - (Y.sum(0) ** 2) / n
    ssw = np.where(ssw < 1e-12, np.nan, ssw)
    F = (ssb / (ncls - 1.0)) / (ssw / (n - ncls))
    return np.nan_to_num(F, nan=0.0)


def select_topk(counts, acts, k, ncls=3, w_ts=0.5, w_fa=0.5):
    """数据驱动 topK 选择（时间方差 + 动作 ANOVA）。

    修复 bug C/G：调用方必须保证 counts/acts 只来自【训练组】；
    打乱脑等对照臂要各自在自己的训练组上重新选择，不能套用真实脑的索引。
    """
    counts = np.asarray(counts, dtype=np.float32)
    acts = np.asarray(acts, dtype=np.int64)
    ts = counts.std(0)
    fa = action_F(counts, acts, ncls=int(ncls))
    score = w_ts * rankpct(ts) + w_fa * rankpct(fa)
    return np.argsort(-score)[:int(k)].astype(np.int64)


def select_topk_unsupervised(counts, k, w_ts=0.6, w_mean=0.4):
    """无监督 topK：时间方差 + 平均发放率（不依赖动作标签，bug C 的 FQI 版）。"""
    counts = np.asarray(counts, dtype=np.float32)
    score = w_ts * rankpct(counts.std(0)) + w_mean * rankpct(counts.mean(0))
    return np.argsort(-score)[:int(k)].astype(np.int64)


# --------------------------------------------------------------------------
# 时间窗口（bug B）：离线 build 与在线 step 使用同一实现，逐元素相等。
# --------------------------------------------------------------------------
class FeatureWindow:
    """win 帧窗口 + 可选跨帧差分。

    语义与旧 build() 完全一致：
      slot j 对应绝对帧 i0 = max(0, t - win + 1 + j)；
      差分 = v[i0] - v[i0-1]，i0 == 0 时以前值 0 计算（差分等于首帧值）。
    在线模式（step）与离线模式（build）对同一帧序列产生逐元素相同的特征。
    """

    def __init__(self, idx, win=5, diff=True):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError('idx 必须是非空 1D int64 数组')
        self.idx = idx
        self.win = int(win)
        if self.win < 1:
            raise ValueError('win 必须 >= 1')
        self.diff = bool(diff)
        self.d = int(idx.size)
        self.dim = self.win * self.d * (2 if self.diff else 1)
        self.reset()

    def reset(self):
        """每个 episode 开头必须调用（bug K 的特征侧配合）。"""
        self.hist = []
        self._t = 0

    def push(self, v):
        """推入当前帧（原始 counts 切片），返回当前帧的绝对序号 t（0 起）。"""
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size != self.d:
            raise ValueError(f'帧向量 {v.size} != idx 长度 {self.d}')
        self.hist.append(v)
        if len(self.hist) > self.win + 1:
            self.hist.pop(0)
        t = self._t
        self._t += 1
        return t

    def feat(self, t):
        """t 为当前帧绝对序号；hist 需已包含 0..t 的帧。"""
        if not self.hist:
            raise RuntimeError('hist 为空：请先 push 至少一帧')
        base = max(0, t - self.win)          # hist[0] 对应的绝对帧
        feat = np.zeros(self.dim, dtype=np.float32)
        col = 0
        for j in range(self.win):
            i0 = max(0, t - self.win + 1 + j)
            off = i0 - base
            hv = self.hist[off]
            feat[col:col + self.d] = hv
            col += self.d
            if self.diff:
                hv0 = self.hist[off - 1] if off > 0 else np.zeros(self.d, np.float32)
                feat[col:col + self.d] = hv - hv0
                col += self.d
        return feat

    def step(self, v):
        """在线一帧：push + 特征。与 build 对同一序列的结果逐元素一致。"""
        t = self.push(v)
        return self.feat(t)

    def build(self, counts):
        """离线批量：counts 为 (T, n) 全脑 counts，内部按 idx 切片。

        返回 (T, dim)。实现复用在线路径，保证两种模式一致。
        """
        counts = np.asarray(counts, dtype=np.float32)
        T = counts.shape[0]
        F = np.zeros((T, self.dim), dtype=np.float32)
        self.reset()
        for t in range(T):
            self.push(counts[t, self.idx])
            F[t] = self.feat(t)
        return F


# --------------------------------------------------------------------------
# 指标口径（bug H）：balanced accuracy 对 1/C（已出现类别数），
# ordinary accuracy 对多数类；同时给 confusion matrix 与逐类 recall。
# --------------------------------------------------------------------------
def classification_metrics(Y, pred, ncls):
    """返回完整指标字典，避免「balanced 对多数类」这种不可比口径。"""
    Y = np.asarray(Y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    n = len(Y)
    if n == 0:
        raise ValueError('空标签')
    counts = np.bincount(Y, minlength=ncls)
    obs = [int(c) for c in range(ncls) if counts[c] > 0]
    rec = {}
    for c in range(ncls):
        m = Y == c
        rec[c] = float((pred[m] == c).mean()) if m.sum() > 0 else None
    bal = float(np.mean([rec[c] for c in obs]))
    conf = [[int(((Y == i) & (pred == j)).sum()) for j in range(ncls)]
            for i in range(ncls)]
    return {
        'n_observed_classes': len(obs),
        'observed_classes': obs,
        'balanced_acc': round(bal, 4),
        'balanced_chance': round(1.0 / len(obs), 4),
        'ordinary_acc': round(float((pred == Y).mean()), 4),
        'majority_frac': round(float(counts.max() / n), 4),
        'per_class_recall': {c: (None if rec[c] is None else round(rec[c], 4))
                             for c in range(ncls)},
        'confusion': conf,
    }


# --------------------------------------------------------------------------
# 每 episode 重置脑状态（bug K）：兼容 doom.engine.Brain 与 corridor_rl_q.Brain。
# --------------------------------------------------------------------------
def reset_brain(brain):
    """把脑的动力学状态恢复到构造后的初始值（膜电位、电导、不应期、
    延迟队列、低通亮度、发放计数、活动列表、光标、累计量）。"""
    n = brain.n
    brain.cursor = 0
    brain.v.fill(-52.0)
    brain.g.fill(0.0)
    brain.drive.fill(0.0)
    brain.refractory.fill(0)
    brain.queue.fill(0)
    brain.queue_count.fill(0)
    brain.counts.fill(0)
    brain.luminance.fill(0.0)
    brain.active.fill(0)
    brain.active_flag.fill(0)
    initial = np.unique(np.r_[brain.retina, brain.lamina, brain.sugar])
    brain.active[:len(initial)] = initial
    brain.active_flag[initial] = 1
    brain.nactive[0] = len(initial)
    if hasattr(brain, 'total_spikes'):
        brain.total_spikes = 0
    if hasattr(brain, 'sim_ms'):
        brain.sim_ms = 0.0


# --------------------------------------------------------------------------
# provenance（bug L）：唯一 run id + 源码/WAD/数据哈希 + 配置落盘。
# --------------------------------------------------------------------------
def run_id(prefix='run'):
    ts = time.strftime('%Y%m%d-%H%M%S')
    h = hashlib.sha256(os.urandom(6)).hexdigest()[:10]
    return f'{prefix}-{ts}-{h}'


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_STRONG_HASH_MAX_BYTES = 50 * 1024 * 1024


def hash_or_weak(path):
    """小文件给强哈希；大文件（>50MB）给 size+mtime 弱指纹，避免每次全量重算。"""
    path = Path(path)
    st = path.stat()
    weak = f'size={st.st_size},mtime={int(st.st_mtime)}'
    if st.st_size > _STRONG_HASH_MAX_BYTES:
        return {'weak': weak}
    return {'sha256': sha256_file(path), 'weak': weak}


def git_head(repo):
    try:
        out = subprocess.run(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def write_provenance(out_dir, config, sources=None, extra=None):
    """写 provenance.json。

    config   : 可 json 化的配置 dict（run_id / seeds / 超参 / CLI 等）
    sources  : {名称: 路径或 None}，记录源码/WAD/数据指纹
    extra    : 附加信息（如评测摘要）
    自动附带：argv 命令、python/numpy 版本（§8.5 第 7 条「记录命令」）。
    """
    import sys
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        'run_id': config.get('run_id'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'argv': [str(x) for x in sys.argv],
        'versions': {'python': sys.version.split()[0],
                     'numpy': np.__version__,
                     'torch': None},
        'config': config,
        'source_hashes': {},
        'extra': extra or {},
    }
    try:
        import torch as _torch
        rec['versions']['torch'] = _torch.__version__
    except Exception:
        pass
    for name, p in (sources or {}).items():
        if p is None:
            rec['source_hashes'][name] = None
            continue
        p = Path(p)
        if p.exists():
            try:
                rec['source_hashes'][name] = hash_or_weak(p)
            except Exception as e:  # 不因哈希失败中断主流程
                rec['source_hashes'][name] = {'error': str(e)}
        else:
            rec['source_hashes'][name] = None
    (out_dir / 'provenance.json').write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    return rec
