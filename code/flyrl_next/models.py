"""v2 多模态读出；图像只在 forward 时由 uint8 归一化。"""
from dataclasses import asdict, dataclass
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


def positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return int(value)


@dataclass(frozen=True)
class ModelSpec:
    image_shape: tuple | None = None
    vector_dim: int = 0
    actions: int = 2
    architecture: str = "auto"

    def __post_init__(self):
        object.__setattr__(self, "actions", positive_int("actions", self.actions))
        if isinstance(self.vector_dim, (bool, np.bool_)) or int(self.vector_dim) != self.vector_dim or self.vector_dim < 0:
            raise ValueError("vector_dim 必须是非负整数")
        object.__setattr__(self, "vector_dim", int(self.vector_dim))
        if self.image_shape is not None:
            shape = tuple(positive_int("image_shape", x) for x in self.image_shape)
            if len(shape) != 3:
                raise ValueError("image_shape 必须为 CHW")
            object.__setattr__(self, "image_shape", shape)
        if self.image_shape is None and not self.vector_dim:
            raise ValueError("至少需要 image 或 vector 输入")
        expected = "hybrid" if self.image_shape and self.vector_dim else "cnn" if self.image_shape else "mlp"
        if self.architecture not in ("auto", expected):
            raise ValueError(f"输入模态要求 architecture={expected}")

    @classmethod
    def coerce(cls, spec):
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, Mapping):
            return cls(**spec)
        raise TypeError("spec 必须是 ModelSpec 或 dict")

    def to_dict(self):
        return asdict(self)


def observation(spec, obs, *, batched=False):
    """验证并复制观测；不把浮点图像静默转换为 uint8。"""
    spec = ModelSpec.coerce(spec)
    if not isinstance(obs, Mapping):
        raise ValueError("obs 必须是 image/vector dict")
    expected = ({"image"} if spec.image_shape else set()) | ({"vector"} if spec.vector_dim else set())
    if set(obs) != expected:
        raise ValueError(f"obs 字段必须为 {sorted(expected)}")
    result, size = {}, None
    for key in expected:
        raw = obs[key]
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().numpy()
        value = np.asarray(raw)
        shape = spec.image_shape if key == "image" else (spec.vector_dim,)
        if batched:
            if value.ndim != len(shape) + 1 or value.shape[1:] != shape or len(value) == 0:
                raise ValueError(f"{key} batch 形状错误")
            if size is not None and size != len(value):
                raise ValueError("观测各模态 batch 长度必须一致")
            size = len(value)
        elif value.shape != shape:
            raise ValueError(f"{key} 形状必须为 {shape}")
        if key == "image":
            if value.dtype != np.uint8:
                raise ValueError("image 必须为 uint8")
        else:
            value = value.astype(np.float32)
            if not np.isfinite(value).all():
                raise ValueError("vector 必须有限")
        result[key] = value.copy()
    return result


class Encoder(nn.Module):
    """MLP 256/128，小型 CNN，以及拼接两种编码的 hybrid。"""
    def __init__(self, spec):
        super().__init__()
        self.spec = ModelSpec.coerce(spec)
        width = 0
        if self.spec.image_shape:
            self.image = nn.Sequential(
                nn.Conv2d(self.spec.image_shape[0], 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(128, 128), nn.ReLU())
            width += 128
        if self.spec.vector_dim:
            self.vector = nn.Sequential(nn.Linear(self.spec.vector_dim, 256), nn.ReLU(),
                                        nn.Linear(256, 128), nn.ReLU())
            width += 128
        self.fusion = nn.Sequential(nn.Linear(width, 128), nn.ReLU()) if width > 128 else nn.Identity()
        self.output_dim = 128

    def forward(self, obs):
        device = next(self.parameters()).device
        features = []
        if self.spec.image_shape:
            image = torch.as_tensor(obs["image"], device=device)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            features.append(self.image(image.float() / 255.))
        if self.spec.vector_dim:
            vector = torch.as_tensor(obs["vector"], dtype=torch.float32, device=device)
            if vector.ndim == 1:
                vector = vector.unsqueeze(0)
            features.append(self.vector(vector))
        return self.fusion(torch.cat(features, dim=-1))


class QNetwork(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = Encoder(spec)
        self.head = nn.Linear(128, self.encoder.spec.actions)

    def forward(self, obs):
        return self.head(self.encoder(obs))


class ActorCritic(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = Encoder(spec)
        self.policy = nn.Linear(128, self.encoder.spec.actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        features = self.encoder(obs)
        return self.policy(features), self.critic(features).squeeze(-1)
