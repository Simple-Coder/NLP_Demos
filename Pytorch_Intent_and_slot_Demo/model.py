"""
Created by xiedong
@Date: 2023/4/17 10:50
"""
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertForIntentClassificationAndSlotFilling(nn.Module):
    def __init__(self, config):
        # 配置
        self.config = config
        # 加载预训练bert
        self.bert = BertModel.from_pretrained(config.bert_dir)
        self.bert_config = self.bert.config

        # 意图识别标签数
        self.num_intent_labels = config.intent_num_labels
        # 槽位填充标签数
        self.slot_num_labels = config.slot_num_labels

        # 意图分类
        self.intent_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_intent_labels)
        )
        # 槽位分类
        self.slot_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.slot_num_labels)
        )

        # CRF 层
        if config.use_crf:
            self.crf = CRF(num_tags=self.slot_num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 喂入模型
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)

        token_output = bert_output[0]  # 每个字对应id
        seq_output = bert_output[1]  # [CLS]

        # 分类
        intent_output = self.intent_classifier(seq_output)
        slot_ouput = self.slot_classifier(token_output)

        return intent_output, slot_ouput
