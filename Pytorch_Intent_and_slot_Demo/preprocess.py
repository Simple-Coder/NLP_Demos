"""
Created by xiedong
@Date: 2023/4/9 20:17
"""
import os.path


class InputExample:
    def __init__(self, set_type, text, intent_label, slot_labels):
        self.set_type = set_type
        self.text = text
        self.intent_label = intent_label
        self.slot_labels = slot_labels


class InputFeature:
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 intent_label_ids,
                 slot_label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_ids = intent_label_ids
        self.slot_label_ids = slot_label_ids


class Processor:
    @classmethod
    def _read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def get_examples(self, args, set_type):
        examples = []
        # 原始文本
        texts_path = os.path.join(args.data_path, args.task, set_type, args.data_input_text_file)
        # 意图
        intents_path = os.path.join(args.data_path, args.task, set_type, args.data_intent_label_file)
        # 槽位
        slots_path = os.path.join(args.data_path, args.task, set_type, args.data_slot_labels_file)

        text_data = self._read_file(texts_path)
        intents_data = self._read_file(intents_path)
        slots_data = self._read_file(slots_path)

        for i, (text, intent, slot) in enumerate(zip(text_data, intents_data, slots_data)):
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = args.intent_labels.index(
                intent) if intent in args.intent_labels else args.intent_labels.index(
                "UNK")
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    args.slot_labels.index(s) if s in args.slot_labels else args.slot_labels.index("UNK"))

            assert len(words) == len(slot_labels)
            examples.append(InputExample(
                set_type,
                text=words,
                intent_label=intent_label,
                slot_labels=slot_labels))
        return examples


def get_features(raw_examples, tokenizer, args):
    features = []
    for i, example in enumerate(raw_examples):
        feature = convert_example_to_feature(i, example, tokenizer, args)
        features.append(feature)

    return features


def convert_example_to_feature(ex_idx, example, tokenizer, args):
    set_type = example.set_type
    text = example.text
    intent_label = example.intent_label
    slot_labels = example.slot_labels

    # 超长截断
    special_token_size = 2  # 2个特殊字符：CLS、SEP
    if len(slot_labels) > args.max_len - special_token_size:
        slot_labels = slot_labels[:(args.max_len - special_token_size)]

    padding_len = args.max_len - len(slot_labels)
    slot_labels_ids = [0] + slot_labels + [0] + ([0] * padding_len)

    inputs = tokenizer.encode_plus(
        text=text,
        max_length=args.max_len,
        padding='max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    intent_label_ids = int(intent_label)
    if ex_idx < 3:
        print(f'*** {set_type}_example-{ex_idx} ***')
        print(f'text: {text}')
        print(f'input_ids: {input_ids}')
        print(f'attention_mask: {attention_mask}')
        print(f'token_type_ids: {token_type_ids}')
        print(f'intent_label_ids: {intent_label_ids}')
        print(f'slot_labels_ids: {slot_labels_ids}')

    return InputFeature(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        intent_label_ids=intent_label_ids,
        slot_label_ids=slot_labels_ids
    )


if __name__ == '__main__':
    from Pytorch_Intent_and_slot_Demo.config import Args
    from transformers import BertTokenizer

    args = Args()
    processor = Processor()
    examples = processor.get_examples(args, 'train')
    print()
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    features = get_features(examples, tokenizer, args)
    print()
