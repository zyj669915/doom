"""全边冻结 MaleCNS 动力学；不含优化器、可塑性或游戏状态输入。"""
import math
import time
from typing import NamedTuple

import numpy as np
from numba import njit


@njit(cache=True)
def advance_th(ptr, post, weight, v, g, refractory, drive, queue, queue_count,
               cursor, steps, dt, counts, active, active_flag, nactive, threshold):
    """逐句抽取 readout_experiment.advance_th；只把 thresh 参数名展开。

    调度继承 doom.engine.advance：积分/阈值 → 延迟投递 → 复位；不应期不接收电导。
    本函数是本包唯一新增的模拟内核，训练和诊断均复用；绝不导入原实验脚本。
    """
    av = math.exp(-dt / 20)
    ag = math.exp(-dt / 5)
    coupling = (av - ag) / 3
    delay_slots = queue.shape[0]
    for step in range(steps):
        slot = cursor % delay_slots
        for k in range(nactive[0]):
            i = active[k]
            if refractory[i] > 0:
                refractory[i] -= 1
            if refractory[i] == 0:
                v[i] = -52 + (v[i] + 52) * av + drive[i] * (1 - av) + g[i] * coupling
                g[i] *= ag
                if v[i] > threshold:
                    counts[i] += 1
                    future = (cursor + int(round(1.8 / dt))) % delay_slots
                    queue[future, queue_count[future]] = i
                    queue_count[future] += 1
        for q in range(queue_count[slot]):
            i = queue[slot, q]
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                if refractory[j] > 0:
                    continue
                g[j] += weight[e]
                if active_flag[j] == 0:
                    active_flag[j] = 1
                    active[nactive[0]] = j
                    nactive[0] += 1
        queue_count[slot] = 0
        future = (cursor + int(round(1.8 / dt))) % delay_slots
        for q in range(queue_count[future]):
            i = queue[future, q]
            v[i] = -52
            g[i] = 0
            refractory[i] = int(round(2.2 / dt))
        cursor += 1
    return cursor


def _edge_stats(ptr, post):
    """分块整数键统计自环和超出唯一有向对的重复边；不创建逐边 Python 对象。"""
    n = len(ptr) - 1
    self_edges = duplicates = 0
    for lo in range(0, n, 4096):
        hi = min(n, lo + 4096)
        a, b = int(ptr[lo]), int(ptr[hi])
        source = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(ptr[lo:hi + 1]))
        target = post[a:b]
        self_edges += int(np.count_nonzero(source == target))
        keys = source * np.int64(n) + target
        duplicates += len(keys) - len(np.unique(keys))
    return {"self_edges": self_edges, "duplicate_edges": duplicates}


def rewire_posts(ptr, post, weight, seed=0):
    """在突触前符号层内随机交换目标端点，保留多重图入/出度和每目标正负入度。

    CSR 源和每条边的权重位置不动；允许新增自环和重复边，不声称保持 simple graph。
    源符号由该源的固定权重推断；混合符号源会报错而非伪称守恒。
    """
    ptr, post, weight = np.asarray(ptr), np.asarray(post), np.asarray(weight)
    n = len(ptr) - 1
    if (ptr.ndim != 1 or post.ndim != 1 or weight.shape != post.shape or n < 1
            or ptr.dtype.kind not in "iu" or post.dtype.kind not in "iu"
            or ptr[0] != 0 or ptr[-1] != len(post) or np.any(np.diff(ptr) < 0)
            or np.any(post < 0) or np.any(post >= n) or not np.isfinite(weight).all()):
        raise ValueError("无效 CSR 重连输入")
    degree = np.diff(ptr)
    source_sign = np.zeros(n, dtype=np.int8)
    nonempty = np.flatnonzero(degree)
    if len(post):
        # 跳过空行后 reduceat 恰好覆盖每个非空源的所有出边。
        signs = np.sign(weight).astype(np.int8)
        lower = np.minimum.reduceat(signs, ptr[nonempty])
        upper = np.maximum.reduceat(signs, ptr[nonempty])
        if np.any(lower != upper):
            raise ValueError("源含混合权重符号；无法以 presynaptic sign 分层")
        source_sign[nonempty] = lower
    edge_sign = np.repeat(source_sign, degree)
    result = post.copy()
    rng = np.random.default_rng(seed)
    for sign in (-1, 0, 1):
        indices = np.flatnonzero(edge_sign == sign)
        endpoints = result[indices]
        rng.shuffle(endpoints)
        result[indices] = endpoints
    before, after = _edge_stats(ptr, post), _edge_stats(ptr, result)
    report = {"seed": int(seed), "edges": len(post), "method": "post permutation within presynaptic sign",
              "preserves_simple_graph": False, "before": before, "after": after,
              "delta": {key: after[key] - before[key] for key in before},
              "changed_endpoints": int(np.count_nonzero(result != post)),
              "in_out_degree_preserved": True, "per_target_sign_degree_preserved": True}
    return result, report


class BrainStep(NamedTuple):
    counts: np.ndarray
    actual_ms: float
    wall_s: float


class FrozenBrain:
    """复用 doom.engine.Brain 的原生数组验证/状态布局；安全加载，不依赖 torch。

    构造与 reset 无隐式 warmup。baseline 显式传 scale=1,bias=12,threshold=-45。
    所有权重只在加载时统一缩放一次，此后标为只读，不进行学习。
    """

    def __init__(self, graph_path, scale=1.5, bias=6.0, threshold=-48.0, dt=.1,
                 *, rewired=False, rewire_seed=0):
        if dt != .1:
            raise ValueError("经过核对的内核只支持 dt=0.1ms")
        if not all(math.isfinite(x) for x in (scale, bias, threshold)) or scale <= 0:
            raise ValueError("scale 必须为有限正数，bias/threshold 必须有限")
        self.graph_path = str(graph_path)
        # 从 doom.engine.Brain.__init__ 复用验证和布局，显式关闭 pickle 并及时关闭归档。
        with np.load(graph_path, allow_pickle=False) as arrays:
            for key in ("ptr", "post", "weight", "ids", "retina", "uv", "lamina", "sugar", "superclass"):
                setattr(self, key, arrays[key])
        self.n = n = len(self.ids)
        for key, dtype in (("ptr", np.int64), ("post", np.int32), ("weight", np.float32),
                           ("ids", np.int64), ("retina", np.int32), ("lamina", np.int32), ("sugar", np.int32)):
            value = getattr(self, key)
            if value.ndim != 1 or value.dtype != dtype or not value.flags.c_contiguous:
                raise ValueError(f"无效原生 graph 数组: {key}")
        if (n < 1 or self.ptr.shape != (n + 1,) or self.ptr[0] != 0 or self.ptr[-1] != len(self.post)
                or np.any(np.diff(self.ptr) < 0) or len(self.weight) != len(self.post)):
            raise ValueError("无效 CSR graph")
        if not np.isfinite(self.weight).all():
            raise ValueError("权重必须有限")
        for value in (self.post, self.retina, self.lamina, self.sugar):
            if np.any(value < 0) or np.any(value >= n):
                raise ValueError("graph 索引越界")
        if (self.uv.shape != (len(self.retina), 2) or not np.isfinite(self.uv).all()
                or np.any(self.uv < 0) or np.any(self.uv > 1)):
            raise ValueError("无效受体 UV")
        if np.unique(self.ids).size != n:
            raise ValueError("graph.ids 必须唯一")
        self.scale, self.bias, self.threshold, self.dt = float(scale), float(bias), float(threshold), float(dt)
        self.weight *= np.float32(scale)
        if not np.isfinite(self.weight).all():
            raise ValueError("缩放后权重溢出")
        self.rewire_report = None
        if rewired:
            self.post, self.rewire_report = rewire_posts(self.ptr, self.post, self.weight, rewire_seed)
        for key in ("ptr", "post", "weight", "ids", "retina", "uv", "lamina", "sugar", "superclass"):
            getattr(self, key).flags.writeable = False
        self.v = np.full(n, -52, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.drive = np.zeros(n, dtype=np.float32)
        self.refractory = np.zeros(n, dtype=np.int16)
        self.queue = np.zeros((int(round(1.8 / dt)) + 1, n), dtype=np.int32)
        self.queue_count = np.zeros(self.queue.shape[0], dtype=np.int32)
        self.counts = np.zeros(n, dtype=np.int32)
        self.luminance = np.zeros(len(self.retina), dtype=np.float32)
        self.active = np.zeros(n, dtype=np.int32)
        self.active_flag = np.zeros(n, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)
        self.reset()

    def reset(self, *, warmup_ms=0.0, warmup_luminance=None):
        """清空全部动态状态和时间余数；可选显式 warmup 计入 sim_ms。"""
        warmup_ms = float(warmup_ms)
        if not math.isfinite(warmup_ms) or warmup_ms < 0:
            raise ValueError("warmup_ms 必须是有限非负数")
        self.cursor = 0
        self.v.fill(-52)
        for key in ("g", "drive", "refractory", "queue", "queue_count", "counts", "luminance", "active", "active_flag"):
            getattr(self, key).fill(0)
        initial = np.unique(np.r_[self.retina, self.lamina, self.sugar])
        self.active[:len(initial)] = initial
        self.active_flag[initial] = 1
        self.nactive[0] = len(initial)
        self.total_spikes, self.sim_ms, self.remainder_ms, self.requested_ms = 0, 0.0, 0.0, 0.0
        if warmup_ms:
            lum = np.zeros(len(self.retina), np.float32) if warmup_luminance is None else warmup_luminance
            return self.step(lum, warmup_ms)
        return None

    def step(self, luminance, duration_ms):
        """返回独立 counts、实际模拟 ms、壁钟秒；余数累计使 35Hz 长期无漂移。"""
        start = time.perf_counter()
        lum = np.asarray(luminance, dtype=np.float32)
        duration_ms = float(duration_ms)
        if lum.shape != self.luminance.shape or not np.isfinite(lum).all():
            raise ValueError("每个受体必须提供一个有限亮度")
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError("duration_ms 必须为有限正数")
        pending = self.remainder_ms + duration_ms
        steps = int(math.floor(pending / self.dt + 1e-10))
        actual_ms = steps * self.dt
        self.remainder_ms = max(0.0, pending - actual_ms)
        self.requested_ms += duration_ms
        self.counts.fill(0)
        if steps:
            alpha = -math.expm1(-actual_ms / 10)
            self.luminance += alpha * (np.clip(lum, 0, 1) - self.luminance)
            self.drive.fill(0)
            self.drive[self.lamina] = self.bias
            self.drive[self.retina] = 30 * self.luminance / (.02 + self.luminance)
            self.cursor = advance_th(self.ptr, self.post, self.weight, self.v, self.g,
                                     self.refractory, self.drive, self.queue, self.queue_count,
                                     self.cursor, steps, self.dt, self.counts, self.active,
                                     self.active_flag, self.nactive, self.threshold)
            self.total_spikes += int(self.counts.sum())
            self.sim_ms = self.cursor * self.dt
        return BrainStep(self.counts.copy(), actual_ms, time.perf_counter() - start)

    def black_run(self, durations_ms, *, warmup_ms=0.0):
        """独立黑屏诊断：新建状态，不触碰当前轨迹，不接优化器、不改变权重。

        durations_ms 是显式有限可迭代时长；不会隐式启动长仿真。
        返回逐神经元累计 counts 和时间、发放量诊断，不生成控制动作。
        """
        durations = np.asarray(tuple(durations_ms), dtype=np.float64)
        if durations.ndim != 1 or not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError("诊断时长必须是有限正数序列")
        # 只共享不可变连接；复制状态数组而不是重新读全图或触碰当前脑状态。
        import copy
        probe = copy.copy(self)
        for key in ("v", "g", "drive", "refractory", "queue", "queue_count", "counts", "luminance", "active", "active_flag", "nactive"):
            setattr(probe, key, np.empty_like(getattr(self, key)))
        probe.reset(warmup_ms=warmup_ms)
        lum = np.zeros(len(probe.retina), dtype=np.float32)
        total = np.zeros(probe.n, dtype=np.int64)
        actual_ms = wall_s = 0.0
        for duration in durations:
            result = probe.step(lum, duration)
            total += result.counts
            actual_ms += result.actual_ms
            wall_s += result.wall_s
        return {"counts": total, "actual_ms": actual_ms, "wall_s": wall_s,
                "frames": len(durations), "warmup_ms": float(warmup_ms),
                "warmup_actual_ms": probe.sim_ms - actual_ms,
                "total_spikes": int(total.sum()), "weights_trained": False}
