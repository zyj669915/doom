"""三组同维双时间尺度表征；RGB以外的信息不进入策略。"""
import hashlib
from pathlib import Path

import numpy as np

from flyrl_next.encoding import Pipeline as BasePipeline, _Frontend, _rgb
from .config import ROOT, PROTOCOL


class Pipeline(BasePipeline):
    """复用冻结校准/安全状态格式，显式绑定arm与有效连接图。"""
    def __init__(self, config, run_dir=None):
        self.c = dict(config)
        self.arm = config["arm"]
        self.route = "A2"
        self.frontend = config["frontend"]
        self.intervention = "none"
        self.seed = config["projection_seed"]
        self.duration_ms = 1000 / 35
        self.vector_dim = self.dim = 1536
        self.image_shape = None
        self.spec = {"vector_dim": 1536, "image_shape": None}
        self.run_dir = None if run_dir is None else Path(run_dir)
        graph = Path(config["graph"])
        graph = graph if graph.is_absolute() else ROOT / graph
        self.brain = None
        if self.arm == "frontend":
            with np.load(graph, allow_pickle=False) as arrays:
                uv = arrays["uv"]
                ids = arrays["ids"][arrays["retina"]]
            bound = {"ids": ids, "uv": uv}
        else:
            from flyrl.brain import FrozenBrain
            self.brain = FrozenBrain(graph, scale=config["scale"], bias=config["bias"],
                                     threshold=config["threshold"], rewired=self.arm == "rewired",
                                     rewire_seed=config["rewire_seed"])
            uv, ids = self.brain.uv, self.brain.ids
            bound = {k: getattr(self.brain, k) for k in ("ids", "ptr", "post", "weight", "retina", "uv", "lamina", "sugar")}
        self.vision = _Frontend(uv, self.frontend, "none", self.seed)
        self.rates = None
        self.group_signature = None
        self.encoding_config = {"protocol": PROTOCOL, "arm": self.arm, "frontend": self.frontend,
                                "projection_seed": self.seed, "rewire_seed": config["rewire_seed"],
                                "scale": config["scale"], "bias": config["bias"], "threshold": config["threshold"],
                                "frontend_projection": "retinal-ID-signed-hash-512-sqrt-count; luminance-input; no-brain"}
        digest = hashlib.sha256()
        for key, array in bound.items():
            digest.update(key.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
        self.graph_signature = digest.hexdigest()
        hashes = [hashlib.sha256(f"flyrl-next-v1:{self.seed}:{int(i)}".encode()).digest() for i in ids]
        self.hash_bucket = np.array([int.from_bytes(h[:8], "little") % 512 for h in hashes], np.int64)
        self.hash_sign = np.array([1 if h[8] & 1 else -1 for h in hashes], np.float32)
        self.hash_scale = np.sqrt(np.maximum(np.bincount(self.hash_bucket, minlength=512), 1))
        self.mu = np.zeros(1536, np.float32)
        self.sd = np.ones(1536, np.float32)
        self.calibrated = False
        self.calibration_count = 0
        self._cal_mean = np.zeros(1536, np.float64)
        self._cal_m2 = np.zeros(1536, np.float64)
        if self.brain is not None and self.run_dir is not None:
            from flyrl.runner import json_write
            json_write(self.run_dir / "graph-report.json", {
                "arm": self.arm, "nodes": self.brain.n, "edges": len(self.brain.post),
                "effective_graph_signature": self.graph_signature,
                "null": self.brain.rewire_report,
                "null_scope": "粗null；不保入强度、自环及重复边；非确认性H2"})
        self.reset()

    def reset(self):
        super().reset()
        self.vision.reset()

    def tic(self, rgb):
        rgb = _rgb(rgb)
        if self.brain is not None:
            super().tic(rgb)
            return
        lum = self.vision.step(rgb, self.duration_ms)
        projected = np.bincount(self.hash_bucket, weights=lum * self.hash_sign,
                                minlength=512) / self.hash_scale
        alpha = -np.expm1(-self.duration_ms / np.array([50., 200.]))
        self._hash_rates += alpha[:, None] * (projected[None, :] - self._hash_rates)
        fast, slow = np.arcsinh(self._hash_rates)
        self._base = np.concatenate((fast, slow, fast - slow)).astype(np.float32)
        self._tic += 1
        self.last_diagnostic = {"actual_ms": 0., "brain_ms": 0., "brain_wall_s": 0.,
                                "spikes": 0, "spiking_neurons": 0, "integration_set": 0}
