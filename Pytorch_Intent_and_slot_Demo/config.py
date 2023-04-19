"""
Created by xiedong
@Date: 2023/4/9 20:04
"""
import torch
import logging



class Args:
    task = 'atis'
    data_path = 'data/'
    intent_labels_path = './data/' + task + '/intent_label.txt'
    slot_labels_path = './data/' + task + '/slot_label.txt'
    bert_dir = 'bert-base-uncased'
    save_dir = './checkpoints/'
    load_dir = './checkpoints/model.pt'

    data_input_text_file = 'seq.in'
    data_intent_label_file = 'label'
    data_slot_labels_file = 'seq.out'

    do_train = False
    do_eval = False
    do_test = True
    do_save = True
    do_predict = True
    load_model = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    intent_label2id = {}
    id2_intentlabel = {}
    with open(intent_labels_path, 'r') as fp:
        intent_labels = fp.read().split('\n')
        for i, intent in enumerate(intent_labels):
            intent_label2id[intent] = i
            id2_intentlabel[i] = intent

    slot_label2id = {}
    id2_slotlabel = {}
    with open(slot_labels_path, 'r') as fp:
        slot_labels = fp.read().split('\n')
        for i, label in enumerate(slot_labels):
            slot_label2id[label] = i
            id2_slotlabel[i] = label
    #
    # tmp = ['O']
    # for label in token_labels:
    #     B_label = 'B-' + label
    #     I_label = 'I-' + label
    #     tmp.append(B_label)
    #     tmp.append(I_label)
    # nerlabel2id = {}
    # id2nerlabel = {}
    # for i, label in enumerate(tmp):
    #     nerlabel2id[label] = i
    #     id2nerlabel[i] = label

    hidden_size = 768
    intent_num_labels = len(intent_labels)
    slot_num_labels = len(slot_labels)
    max_len = 50
    batchsize = 64
    lr = 2e-5
    epoch = 10
    hidden_dropout_prob = 0.1
    use_crf = True


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    args = Args()
    print(args.intent_labels)
    print(args.slot_labels)

    print(args.intent_label2id)
    print(args.id2_intentlabel)
    print(args.slot_label2id)
    print(args.id2_slotlabel)

    print("-----intent_num_labels:", args.intent_num_labels)
    print("-----slot_num_labels:", args.slot_num_labels)
