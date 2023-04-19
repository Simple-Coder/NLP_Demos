"""
Created by xiedong
@Date: 2023/4/19 21:46
"""
from transformers import BertConfig
from transformers import BertTokenizer

import logging


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}
