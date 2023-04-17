"""
Created by xiedong
@Date: 2023/4/17 11:20
"""
from preprocess import *
import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
from dataset import *
from model import *

if __name__ == '__main__':
    # 加载配置
    args = Args()
    # 加载bert
    tokenizer = BertModel.from_pretrained(args.bert_dir)

    # 训练
    if args.do_train:
        raw_examples = Processor.get_examples(args.data_path, 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        train_dataset = BertDataset(train_features)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    # model
    model = BertForIntentClassificationAndSlotFilling(args)
    model.to(args.device)

    trainer = Trainer(model, args)

    if args.do_train:
        trainer.train(train_loader)
