"""冻结连接组感觉/神经模块，和下游小读出 RL 训练器保持边界。"""
from .encoding import Groups, MultiscaleRates, Vision, build_groups

__all__ = ["Vision", "Groups", "build_groups", "MultiscaleRates", "FrozenBrain",
           "BrainStep", "rewire_posts"]


def __getattr__(name):
    # 只使用视觉/特征时，不提前加载 numba 或任何游戏依赖。
    if name in ("FrozenBrain", "BrainStep", "rewire_posts"):
        from . import brain
        value = getattr(brain, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
