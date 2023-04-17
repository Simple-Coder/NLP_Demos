"""
Created by xiedong
@Date: 2023/4/17 12:23
"""


class IntentAndSlotExecutor:
    def __init__(self, model, config, train_dataset):
        self.model = model
        self.device = config.device
        self.config = config
        self.epoch = config.epoch

        self.train_dataset = train_dataset

        # model to device（CPU or GPU）
        self.model.to(self.device)

    def train(self):
        pass
