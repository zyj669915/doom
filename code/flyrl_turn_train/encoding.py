"""复用完整脑/匹配frontend编码，独立绑定短任务与输入干预。"""
from flyrl_stage.encoding import Pipeline as StagePipeline
from .config import PROTOCOL, ENV_PROTOCOL


class Pipeline(StagePipeline):
    def __init__(self, config, run_dir=None):
        super().__init__(config, run_dir)
        self.encoding_config.update(protocol=PROTOCOL, environment_protocol=ENV_PROTOCOL)

    def set_intervention(self, mode):
        if mode not in ('none', 'black'):
            raise ValueError('本轮只允许正常/黑屏冻结干预')
        # 从episode开始应用，不重做校准、不改快照中的训练编码配置。
        self.intervention = mode
        self.vision.vision.intervention = mode
        self.reset()
