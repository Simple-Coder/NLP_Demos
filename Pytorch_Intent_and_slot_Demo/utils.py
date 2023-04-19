"""
Created by xiedong
@Date: 2023/4/19 21:46
"""
from transformers import BertConfig
from transformers import BertTokenizer

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}
