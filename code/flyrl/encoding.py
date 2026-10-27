"""无动作标签的视觉前端、预注册群体和实际时间尺度特征。"""
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np


CELL_TYPES = ("L1", "L2", "L3", "T4", "T5", "LC", "LPLC", "DNa02",
              "DNp09", "MDN", "DNp20", "DNpe017", "MN9")
SUPERCLASSES = ("ol_intrinsic", "cb_intrinsic", "vnc_intrinsic", "visual_projection",
                "vnc_sensory", "ol_sensory", "cb_sensory", "ascending_neuron",
                "descending_neuron", "vnc_motor", "visual_centrifugal", "sensory_ascending",
                "cb_motor", "vnc_efferent", "cb_endocrine", "ENS", "vnc_tbc",
                "vnc_sensory_tbc", "vnc_endocrine", "cb_sensory_tbc", "sensory_descending",
                "efferent_ascending", "efferent_descending", "cb_efferent",
                "visual_projection_tbc", "sensory_ascending_tbc", "descending_neuron_tbc",
                "other")
SIDES = ("L", "R", "unknown")


def _retinal_samples(rgb, uv):
    """复制 doom.game.retinal_samples 的双线性线性光亮度公式，避免导入 vizdoom。"""
    h, w = rgb.shape[:2]
    x, y = uv[:, 0] * (w - 1), uv[:, 1] * (h - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0

    def luma(pixels):
        p = pixels.astype(np.float32) / 255
        p = np.where(p <= .04045, p / 12.92, ((p + .055) / 1.055) ** 2.4)
        return p @ np.asarray([.2126, .7152, .0722], dtype=np.float32)

    return ((1-dx)*(1-dy)*luma(rgb[y0, x0]) + dx*(1-dy)*luma(rgb[y0, x1])
            + (1-dx)*dy*luma(rgb[y1, x0]) + dx*dy*luma(rgb[y1, x1])).astype(np.float32)


class Vision:
    """RGB24 → receptor luminance；scramble 是固定受体置换，不重排时间。"""

    def __init__(self, uv, frontend="raw", intervention="none", seed=0, sampler=None):
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all() or np.any((uv < 0) | (uv > 1)):
            raise ValueError("uv 必须是 [0,1] 内的有限 (R,2) 数组")
        if frontend not in ("raw", "agc_motion") or intervention not in ("none", "black", "scramble"):
            raise ValueError("未知 frontend 或 intervention")
        self.uv = uv.copy()
        self.frontend, self.intervention = frontend, intervention
        # 可显式传入 doom.game.retinal_samples；默认不导入任何游戏依赖。
        self.sampler = _retinal_samples if sampler is None else sampler
        self.permutation = np.random.default_rng(seed).permutation(len(uv))
        self.reset()

    def reset(self):
        """清除运动历史；固定空间置换跨 episode 不变。"""
        self.previous = None

    def step(self, rgb, intervention=None):
        mode = self.intervention if intervention is None else intervention
        if mode not in ("none", "black", "scramble"):
            raise ValueError("未知输入干预")
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1 or rgb.dtype != np.uint8:
            raise ValueError("RGB 输入必须是非空 H×W×3 uint8 RGB24")
        # 黑屏必须先于运动/AGC；显式压零避免切换黑屏时产生伪瞬变。
        lum = np.zeros(len(self.uv), np.float32) if mode == "black" else np.asarray(self.sampler(rgb, self.uv), dtype=np.float32)
        if lum.shape != (len(self.uv),) or not np.isfinite(lum).all():
            raise ValueError("采样器必须返回每个受体一个有限亮度")
        if mode == "scramble":
            lum = lum[self.permutation]
        out = lum.copy()
        if self.frontend == "agc_motion" and mode != "black":
            # 与 readout_experiment 的 motion → agc 顺序及常数一致，但不导入该脚本。
            if self.previous is not None:
                out = np.clip(lum + 1.2 * (lum - self.previous), 0, 1)
            sd = out.std()
            if sd >= 1e-4:
                out = .5 + .5 * np.tanh((out - out.mean()) / (sd + 1e-4) / 1.5)
        self.previous = lum.copy()
        return np.clip(out, 0, 1).astype(np.float32)

    __call__ = step


def _ids(values):
    """用精确整数字符串作为连接键，拒绝浮点 ID，防止大整数精度丢失。"""
    result = []
    for value in values:
        if isinstance(value, (float, np.floating, bool, np.bool_)):
            raise ValueError("source_id 不允许浮点或布尔类型")
        text = str(value).strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError(f"无效 source_id: {text!r}")
        result.append(str(int(text)))
    if len(set(result)) != len(result):
        raise ValueError("source_id 必须唯一")
    return result


def _side(value):
    return {"l": "L", "left": "L", "r": "R", "right": "R"}.get(str(value).strip().lower(), "unknown")


@dataclass(frozen=True)
class Groups:
    """固定互斥分组；所有神经元恰好一次，空组不删除。"""
    graph_ids: np.ndarray
    group_names: tuple
    group_indices: tuple
    group_counts: np.ndarray
    annotations: tuple
    provenance: dict

    def save(self, path):
        """无 pickle 保存索引、人数、注释与 ID 顺序，供训练记录复算。"""
        offsets = np.r_[0, np.cumsum(self.group_counts)]
        np.savez_compressed(path, graph_ids=self.graph_ids,
                            group_names=np.asarray(self.group_names),
                            indices=np.concatenate(self.group_indices), offsets=offsets,
                            group_counts=self.group_counts, annotations=np.asarray(self.annotations),
                            provenance=np.asarray(json.dumps(self.provenance, ensure_ascii=False)))


def build_groups(graph_ids, neurons, catalog, *, side_by_id=None):
    """表或 feather 路径 → 123 个固定组；只按 source_id 对齐，绝不按表行号。"""
    # pandas 仅在调用分组时导入；纯视觉和速率路径只需要 numpy。
    import pandas as pd

    def table(value):
        return pd.read_feather(value) if isinstance(value, (str, Path)) else value

    ids = _ids(np.asarray(graph_ids))
    frames = [table(neurons), table(catalog)]
    mappings = []
    for frame in frames:
        if "source_id" not in frame:
            raise ValueError("normalized 表缺少 source_id")
        keys = _ids(frame["source_id"])
        mappings.append(dict(zip(keys, frame.to_dict("records"))))
    missing = [key for key in ids if key not in mappings[0] and key not in mappings[1]]
    if missing:
        raise ValueError(f"有 {len(missing)} 个 graph.ids 缺少注释，例如 {missing[:3]}")
    sides = {}
    if side_by_id is not None:
        keys = list(side_by_id)
        sides = dict(zip(_ids(keys), (side_by_id[key] for key in keys)))
    names = tuple(f"cell_type:{typ}:{side}" for typ in CELL_TYPES for side in SIDES) + tuple(
        f"superclass_residual:{typ}:{side}" for typ in SUPERCLASSES for side in SIDES)
    positions = {name: i for i, name in enumerate(names)}
    members = [[] for _ in names]

    def get(rows, field):
        for row in rows:
            value = row.get(field)
            if value is not None and not pd.isna(value) and str(value).strip():
                return str(value).strip()
        return ""

    for index, key in enumerate(ids):
        # neurons 的细胞类型优先；catalog 的 superclass 优先。
        node, cat = mappings[0].get(key, {}), mappings[1].get(key, {})
        typ = get((node, cat), "cell_type")
        sup = get((cat, node), "superclass")
        side = _side(sides[key]) if key in sides else _side(next(
            (v for field in ("side", "somaSide", "soma_side", "hemisphere")
             if (v := get((node, cat), field))), "unknown"))
        family = next((x for x in CELL_TYPES if typ == x or
                       (x in ("T4", "T5", "LC", "LPLC") and typ.startswith(x))), None)
        name = f"cell_type:{family}:{side}" if family else f"superclass_residual:{sup if sup in SUPERCLASSES else 'other'}:{side}"
        members[positions[name]].append(index)
    indices = tuple(np.asarray(group, dtype=np.int64) for group in members)
    counts = np.asarray([len(group) for group in indices], dtype=np.int64)
    annotations = tuple("预注册细胞类型/家族；未据动作筛选" if name.startswith("cell_type:") else
                        "排除所有已分入细胞类型组的神经元，避免母集重复" for name in names)
    graph_ids = np.asarray([int(key) for key in ids], dtype=np.int64)
    for array in (graph_ids, counts, *indices):
        array.flags.writeable = False
    return Groups(graph_ids, names, indices, counts, annotations,
                  {"version": 1, "alignment": "exact source_id", "partition": "cell_type then superclass residual",
                   "side_policy": "缺失/未知侧别保留 unknown，不推断左右", "action_labels_used": False})


class MultiscaleRates:
    """每组平均 Hz → 50/200ms 因果低通 → log1p；输出按时间尺度排列。"""

    def __init__(self, groups, taus_ms=(50.0, 200.0), rate_scale_hz=1.0):
        if not isinstance(groups, Groups):
            raise TypeError("groups 必须由 build_groups 生成")
        self.groups = groups
        self.group_names, self.group_indices = groups.group_names, groups.group_indices
        self.taus_ms = np.asarray(taus_ms, dtype=np.float64)
        self.rate_scale_hz = float(rate_scale_hz)
        if self.taus_ms.ndim != 1 or not self.taus_ms.size or not np.isfinite(self.taus_ms).all() or np.any(self.taus_ms <= 0):
            raise ValueError("低通时间常数必须为有限正数")
        if not np.isfinite(self.rate_scale_hz) or self.rate_scale_hz <= 0:
            raise ValueError("rate_scale_hz 必须为有限正数")
        self.dim = len(self.group_names) * len(self.taus_ms)
        self.feature_names = tuple(f"log1p_mean_hz:{tau:g}ms:{name}" for tau in self.taus_ms for name in self.group_names)
        self.reset()

    def reset(self):
        self.rates = np.zeros((len(self.taus_ms), len(self.group_names)), dtype=np.float64)
        self.elapsed_ms = 0.0

    def step(self, counts, duration_ms):
        counts = np.asarray(counts)
        duration_ms = float(duration_ms)
        if counts.shape != self.groups.graph_ids.shape or not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("counts 必须是与 graph.ids 对齐的有限非负向量")
        if not np.isfinite(duration_ms) or duration_ms < 0 or (duration_ms == 0 and np.any(counts)):
            raise ValueError("duration_ms 必须是实际非负模拟时间，零时间不能有脉冲")
        if duration_ms > 0:
            hz = np.asarray([counts[idx].sum(dtype=np.float64) / len(idx) if len(idx) else 0
                             for idx in self.group_indices]) * (1000 / duration_ms)
            alpha = -np.expm1(-duration_ms / self.taus_ms)
            self.rates += alpha[:, None] * (hz[None, :] - self.rates)
            self.elapsed_ms += duration_ms
        return np.log1p(self.rates / self.rate_scale_hz).ravel().astype(np.float32)

    __call__ = step
