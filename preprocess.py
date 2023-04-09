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


if __name__ == '__main__':
    from config import Args

    args = Args()
    processor = Processor()
    examples = processor.get_examples(args, 'train')
    print()
