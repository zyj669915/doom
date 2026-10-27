"""逐 tic 冻结全脑编码及视觉对照；不包含游戏、优化器或动作标签。

observation() 是决策边界；重复读取不会追加历史。reset() 只清 episode
动态，不清校准统计或固定 ID hash。state_dict 只保存冻结编码语义和校准统计，
不保存脑/游戏动态，因此不是精确续训快照。
"""
from collections import deque
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATHS = {
    "graph": "doomfly/outputs/doom/malecns_v1/graph.npz",
    "neurons": "doomfly/connectome_data/malecns_v1/normalized/neurons.feather",
    "catalog": "doomfly/connectome_data/malecns_v1/normalized/catalog.feather",
    "annotations": "doomfly/connectome_data/malecns_v1/annotations.feather",
}


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _rgb(value):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("RGB 输入必须是非空 H×W×3 uint8")
    return value


def _gray84(rgb):
    """双线性缩放完整视野，不裁剪；灰度为显示空间 BT.601 亮度。"""
    lum = rgb.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    h, w = lum.shape
    x = np.linspace(0, w - 1, 84)
    y = np.linspace(0, h - 1, 84)
    x0, y0 = x.astype(np.intp), y.astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, (y - y0)[:, None]
    out = ((1 - dy) * ((1 - dx) * lum[y0[:, None], x0] + dx * lum[y0[:, None], x1])
           + dy * ((1 - dx) * lum[y1[:, None], x0] + dx * lum[y1[:, None], x1]))
    return np.rint(out).clip(0, 255).astype(np.uint8)


def _local_contrast_rgb(rgb, window=9):
    """线性光灰度9×9局部Weber对比度；积分图box均值，边缘复制。

    映射到[0,1]线性亮度后重新sRGB编码，交由Vision保留原有线性化与
    双线性UV采样语义。无需空间裁剪，不改变受体位置或读取图注释。
    """
    rgb = _rgb(rgb)
    pixels = rgb.astype(np.float64) / 255.
    linear = np.where(pixels <= .04045, pixels / 12.92,
                      ((pixels + .055) / 1.055) ** 2.4)
    gray = linear @ np.array([.2126, .7152, .0722])
    radius = window // 2
    padded = np.pad(gray, radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    local_mean = (integral[window:, window:] - integral[:-window, window:]
                  - integral[window:, :-window] + integral[:-window, :-window]) / (window * window)
    contrast = .5 + .5 * np.tanh((gray - local_mean) / (.05 + local_mean))
    encoded = np.where(contrast <= .0031308, 12.92 * contrast,
                       1.055 * contrast ** (1 / 2.4) - .055)
    gray8 = np.rint(encoded * 255).clip(0, 255).astype(np.uint8)
    return np.repeat(gray8[:, :, None], 3, axis=2)


class _Frontend:
    """raw/agc_motion 复用 v1；adapt/contrast 在受体输入处真实作用。"""

    def __init__(self, uv, frontend, intervention, seed):
        from flyrl.encoding import Vision
        self.frontend = frontend
        self.vision = Vision(uv, "agc_motion" if frontend == "agc_motion" else "raw",
                             intervention, seed)
        self.reset()

    def reset(self):
        self.vision.reset()
        self.adaptation = None

    def step(self, rgb, duration_ms):
        if self.frontend == "contrast":
            rgb = _local_contrast_rgb(rgb)
        lum = self.vision.step(rgb)
        if self.vision.intervention == "black":
            self.adaptation = np.zeros_like(lum)
            return lum
        if self.frontend == "adapt":
            if self.adaptation is None:
                self.adaptation = lum.copy()
            # 200ms 因果适应基线；首帧保持原亮度，不隐式预热。
            out = np.clip(lum + (lum - self.adaptation), 0, 1)
            self.adaptation += -np.expm1(-duration_ms / 200.) * (lum - self.adaptation)
            return out.astype(np.float32)
        return lum.astype(np.float32)


class Pipeline:
    """A1..A7 共用逐 tic 接口；纯视觉路径不打开图或注释。

    spec 为 {'vector_dim': int, 'image_shape': tuple|None}。A1 为四个
    246维群体历史及对应相邻决策差分；A2/A3/A7 为512维 signed ID hash
    的快、慢、快减慢。A4每tic帧为 gray/ON/OFF，A5/A6/A7为gray。
    图像历史按旧到新排列，不足四帧时左侧补零。
    """

    def __init__(self, config, run_dir=None):
        self.c = dict(config)
        self.route = self.c.get("route", "A1")
        if self.route not in tuple(f"A{i}" for i in range(1, 8)):
            raise ValueError("route 必须是 A1..A7")
        self.frontend = self.c.get("frontend", "raw")
        if self.frontend not in ("raw", "agc_motion", "adapt", "contrast"):
            raise ValueError("未知 frontend")
        self.intervention = self.c.get("intervention", "none")
        if self.intervention not in ("none", "black", "scramble"):
            raise ValueError("未知 intervention")
        self.seed = int(self.c.get("projection_seed", 23))
        self.duration_ms = 1000 / 35
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.vector_dim = 1968 if self.route == "A1" else (1536 if self.route in ("A2", "A3", "A7") else 0)
        self.image_shape = ((12 if self.route == "A4" else 4), 84, 84) if self.route in ("A4", "A5", "A6", "A7") else None
        self.spec = {"vector_dim": self.vector_dim, "image_shape": self.image_shape}
        self.dim = self.vector_dim
        self.brain = self.vision = self.groups = self.rates = None
        self.hash_bucket = np.empty(0, np.int64)
        self.hash_sign = np.empty(0, np.float32)
        self.hash_scale = np.ones(512, np.float64)
        self.group_signature = None
        self.graph_signature = None
        self.side_report = None
        self.encoding_config = {"route": self.route, "frontend": self.frontend,
                                "intervention": self.intervention, "projection_seed": self.seed,
                                "scale": float(self.c.get("scale", 1.5)),
                                "bias": float(self.c.get("bias", 6.)),
                                "threshold": float(self.c.get("threshold", -48.))}
        if self.vector_dim:
            from flyrl.brain import FrozenBrain
            self.brain = FrozenBrain(_resolve(self.c.get("graph", _DEFAULT_PATHS["graph"])),
                                     scale=self.encoding_config["scale"], bias=self.encoding_config["bias"],
                                     threshold=self.encoding_config["threshold"])
            self.vision = _Frontend(self.brain.uv, self.frontend, self.intervention, self.seed)
            # 绑定完整图，包括边/权重/受体布局，防止同ID但不同连接静默复用。
            digest = hashlib.sha256()
            for key in ("ids", "ptr", "post", "weight", "retina", "uv", "lamina", "sugar"):
                array = getattr(self.brain, key)
                digest.update(key.encode())
                digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode())
                digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
            self.graph_signature = digest.hexdigest()
            if self.route == "A1":
                from flyrl.encoding import build_groups, MultiscaleRates
                from flyrl.runner import anatomical_sides
                sides, self.side_report = anatomical_sides(
                    _resolve(self.c.get("annotations", _DEFAULT_PATHS["annotations"])), self.brain.ids)
                self.groups = build_groups(self.brain.ids,
                    _resolve(self.c.get("neurons", _DEFAULT_PATHS["neurons"])),
                    _resolve(self.c.get("catalog", _DEFAULT_PATHS["catalog"])), side_by_id=sides)
                self.groups.provenance.update(side_source=self.side_report)
                self.rates = MultiscaleRates(self.groups)
                group_digest = hashlib.sha256()
                for name, indices in zip(self.groups.group_names, self.groups.group_indices):
                    group_digest.update(name.encode())
                    group_digest.update(indices.tobytes())
                self.group_signature = group_digest.hexdigest()
            else:
                # Python hash 随进程变化，精确整数ID的SHA256不变化，不转float。
                hashes = [hashlib.sha256(f"flyrl-next-v1:{self.seed}:{int(ident)}".encode()).digest()
                          for ident in self.brain.ids]
                self.hash_bucket = np.array([int.from_bytes(x[:8], "little") % 512 for x in hashes], np.int64)
                self.hash_sign = np.array([1 if x[8] & 1 else -1 for x in hashes], np.float32)
                self.hash_scale = np.sqrt(np.maximum(np.bincount(self.hash_bucket, minlength=512), 1))
        self.mu = np.zeros(self.vector_dim, np.float32)
        self.sd = np.ones(self.vector_dim, np.float32)
        self.calibrated = False
        self.calibration_count = 0
        self._cal_mean = np.zeros(self.vector_dim, np.float64)
        self._cal_m2 = np.zeros(self.vector_dim, np.float64)
        self.reset()

    def reset(self):
        """清动态，不清冻结归一化、在线校准累积或固定hash。"""
        if self.brain is not None:
            self.brain.reset()
            self.vision.reset()
        if self.rates is not None:
            self.rates.reset()
        self._hash_rates = np.zeros((2, 512), np.float64)
        self._base = np.zeros(246 if self.route == "A1" else 1536, np.float32)
        self._vectors = deque([np.zeros(246, np.float32) for _ in range(4)], maxlen=4)
        self._differences = deque([np.zeros(246, np.float32) for _ in range(4)], maxlen=4)
        self._last_decision = np.zeros(246, np.float32)
        channels = 3 if self.route == "A4" else 1
        self._images = deque([np.zeros((channels, 84, 84), np.uint8) for _ in range(4)], maxlen=4)
        self._previous_gray = None
        self._tic = 0
        self._decision_tic = -1
        self._raw_vector = None
        self.last_diagnostic = {"brain_ms": 0., "actual_ms": 0., "brain_wall_s": 0.,
                                "spikes": 0, "spiking_neurons": 0, "integration_set": 0}

    def tic(self, rgb):
        """只推进一个35Hz tic，不生成策略决策，不更新归一化统计。"""
        rgb = _rgb(rgb)
        if self.brain is not None:
            result = self.brain.step(self.vision.step(rgb, self.duration_ms), self.duration_ms)
            counts = result.counts
            if self.rates is not None:
                self._base = self.rates.step(counts, result.actual_ms)
            elif result.actual_ms > 0:
                # 先对齐每神经元Hz再signed投影；桶按sqrt人数归一化。
                projected = np.bincount(self.hash_bucket, weights=counts * self.hash_sign,
                                       minlength=512) * (1000 / result.actual_ms) / self.hash_scale
                alpha = -np.expm1(-result.actual_ms / np.array([50., 200.]))
                self._hash_rates += alpha[:, None] * (projected[None, :] - self._hash_rates)
                fast, slow = np.arcsinh(self._hash_rates)
                self._base = np.concatenate((fast, slow, fast - slow)).astype(np.float32)
            self.last_diagnostic = {"brain_ms": float(result.actual_ms), "actual_ms": float(result.actual_ms),
                "brain_wall_s": float(result.wall_s), "spikes": int(counts.sum(dtype=np.int64)),
                "spiking_neurons": int(np.count_nonzero(counts)), "integration_set": int(self.brain.nactive[0])}
        if self.image_shape is not None:
            gray = _gray84(rgb)
            if self.intervention == "black":
                gray.fill(0)
            elif self.intervention == "scramble":
                permutation = np.random.default_rng(self.seed).permutation(gray.size)
                gray = gray.ravel()[permutation].reshape(gray.shape)
            if self.route == "A4":
                difference = np.zeros_like(gray, np.int16) if self._previous_gray is None else gray.astype(np.int16) - self._previous_gray
                frame = np.stack((gray, np.maximum(difference, 0).astype(np.uint8),
                                  np.maximum(-difference, 0).astype(np.uint8)))
            else:
                frame = gray[None, :, :]
            self._images.append(frame.copy())
            self._previous_gray = gray.astype(np.int16)
        self._tic += 1

    def raw_vector(self):
        """当前决策边界未归一化特征；同tic重复调用幂等。"""
        if not self.vector_dim:
            return np.empty(0, np.float32)
        if self._decision_tic != self._tic:
            if self.route == "A1":
                self._vectors.append(self._base.copy())
                self._differences.append(self._base - self._last_decision)
                self._last_decision = self._base.copy()
                self._raw_vector = np.concatenate((*self._vectors, *self._differences)).astype(np.float32)
            else:
                self._raw_vector = self._base.copy()
            self._decision_tic = self._tic
        return self._raw_vector.copy()

    def observation(self):
        result = {}
        if self.vector_dim:
            result["vector"] = np.clip((self.raw_vector() - self.mu) / self.sd, -10, 10).astype(np.float32)
        if self.image_shape is not None:
            result["image"] = np.concatenate(tuple(self._images), axis=0).copy()
        return result

    def calibration_update(self, vector=None):
        """训练随机轨迹上每决策调用一次Welford；freeze后拒绝意外更新。"""
        if self.calibrated:
            raise RuntimeError("归一化已冻结；调用start_calibration显式开启新校准")
        if not self.vector_dim:
            return 0
        value = self.raw_vector() if vector is None else np.asarray(vector, np.float64)
        if value.shape != (self.vector_dim,) or not np.isfinite(value).all():
            raise ValueError("校准特征维度错误或非有限")
        self.calibration_count += 1
        delta = value - self._cal_mean
        self._cal_mean += delta / self.calibration_count
        self._cal_m2 += delta * (value - self._cal_mean)
        return self.calibration_count

    def start_calibration(self):
        """显式清校准累积；reset从不调用本方法。"""
        self.calibrated = False
        self.calibration_count = 0
        self._cal_mean.fill(0)
        self._cal_m2.fill(0)
        self.mu.fill(0)
        self.sd.fill(1)

    def freeze_calibration(self):
        if self.vector_dim and self.calibration_count < 2:
            raise ValueError("冻结校准至少需要两个训练决策样本")
        if self.vector_dim:
            self.mu = self._cal_mean.astype(np.float32)
            self.sd = np.maximum(np.sqrt(np.maximum(self._cal_m2 / self.calibration_count, 0)), .01).astype(np.float32)
        self.calibrated = True

    def state_dict(self):
        """只返回weights_only安全的tensor/标量/list/dict，不含numpy或脑动态。"""
        import torch
        return {"schema": 1, "feature_version": "flyrl-next-encoding-v3-local-contrast-asinh", "config": dict(self.encoding_config),
            "graph_signature": self.graph_signature, "group_signature": self.group_signature,
            "mu": torch.from_numpy(self.mu.copy()), "sd": torch.from_numpy(self.sd.copy()),
            "calibrated": self.calibrated, "calibration_count": self.calibration_count,
            "calibration_mean": torch.from_numpy(self._cal_mean.copy()),
            "calibration_m2": torch.from_numpy(self._cal_m2.copy()),
            "hash_bucket": torch.from_numpy(self.hash_bucket.copy()),
            "hash_sign": torch.from_numpy(self.hash_sign.copy()),
            "hash_scale": torch.from_numpy(self.hash_scale.copy())}

    def load_state_dict(self, state):
        """验证所有语义及统计后原子加载；不接受任意pickle对象或脑动态。"""
        import torch
        if (state.get("schema") != 1 or state.get("feature_version") != "flyrl-next-encoding-v3-local-contrast-asinh"
                or state.get("config") != self.encoding_config
                or state.get("graph_signature") != self.graph_signature
                or state.get("group_signature") != self.group_signature):
            raise ValueError("checkpoint 编码语义/配置/全图/群体不匹配")
        def array(key, shape):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise ValueError(f"checkpoint {key} 必须为稠密tensor")
            result = value.detach().cpu().numpy().copy()
            if result.shape != shape or not np.isfinite(result).all():
                raise ValueError(f"checkpoint {key} 形状或有限性错误")
            return result
        mu = array("mu", (self.vector_dim,)).astype(np.float32)
        sd = array("sd", (self.vector_dim,)).astype(np.float32)
        mean = array("calibration_mean", (self.vector_dim,)).astype(np.float64)
        m2 = array("calibration_m2", (self.vector_dim,)).astype(np.float64)
        count, calibrated = state.get("calibration_count"), state.get("calibrated")
        if (type(count) is not int or count < 0 or type(calibrated) is not bool or np.any(sd < .01)
                or not np.isfinite(mu).all() or not np.isfinite(sd).all()
                or np.any(m2 < -1e-8) or (self.vector_dim and calibrated and count < 2)):
            raise ValueError("checkpoint 校准统计无效")
        for key in ("hash_bucket", "hash_sign", "hash_scale"):
            expected = getattr(self, key)
            if not np.array_equal(array(key, expected.shape), expected):
                raise ValueError(f"checkpoint {key} 与精确ID hash不匹配")
        self.mu, self.sd = mu, sd
        self._cal_mean, self._cal_m2 = mean, m2
        self.calibration_count, self.calibrated = count, calibrated
        self.reset()
