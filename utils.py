# -*- coding: utf-8 -*-
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained("./pre_train")
CLS, PAD = '[CLS]', '<PAD>'


def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


class BaseDataset(Dataset):
    def __init__(self, features, mode):
        # (token_ids, attention_masks, token_type_ids, lable, tigger_loc)
        self.nums = len(features)

        self.token_ids = [torch.tensor(example[0]).long() for example in features]
        self.attention_masks = [torch.tensor(example[1]).float() for example in features]
        self.token_type_ids = [torch.tensor(example[2]).long() for example in features]

        self.labels = None
        if mode == 'train':
            self.labels = [torch.tensor(example[3]) for example in features]

    def __len__(self):
        return self.nums


class Datasets(BaseDataset):
    def __init__(self, features,
                 mode,
                 use_trigger_distance=False):
        super(Datasets, self).__init__(features, mode)
        self.trigger_distance = None
        self.trigger_label = [torch.tensor([example[4][0][1], example[4][0][2]]).long() for example in features]
        if use_trigger_distance:
            self.trigger_distance = [torch.tensor(example[5]).long()
                                     for example in features]

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index],
                'trigger_index': self.trigger_label[index]}

        if self.trigger_distance is not None:
            data['trigger_distance'] = self.trigger_distance[index]

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data


class SOParse:
    @staticmethod
    def read_jsons(file_path):
        _file = open(file_path, "r", encoding="utf-8").readlines()
        return _file


class HANSPO(SOParse):
    @staticmethod
    def _example_generator(raw_examples, set_type, max_seq_len=256):

        def _relative_distan_tigger(text, tigger_loc):
            tigger_loc_start = tigger_loc[0][1]
            tigger_loc_end = tigger_loc[0][2]
            tigger_distance_ = [0] * (len(text) + 2)
            for i in range(tigger_loc_start + 1, 0, -1):
                tigger_distance_[tigger_loc_start + 1 - i] = i
            for i in range(tigger_loc_end + 1, len(tigger_distance_)):
                tigger_distance_[i] = i + 1 - tigger_loc_end - 1
            return tigger_distance_

        examples = []
        callback_info = []
        type_nums = 0
        pad_lables = [[0] * 4]
        type_weight = {'object': 0,
                       'subject': 0, }
        for _e in raw_examples:
            tigger_loc = []
            _e = json.loads(_e)
            text = _e.get("data")
            text_ = [i for i in text]
            text_ = text_
            lable = [[0] * (len(type_weight) * 2) for i in range(max_seq_len)]
            data_lables = _e.get("label")
            for s in data_lables:
                if s[2] == "触发词":
                    tigger_loc.append([text[s[0]:s[1]], s[0], s[1]])  # 原始位置不需要改变
                # 主体客体
                if s[2] == "主体":
                    lable[s[0] + 1][0] = 1
                    lable[s[1]][1] = 1
                if s[2] == "客体":
                    lable[s[0] + 1][2] = 1
                    lable[s[1]][3] = 1
            tigger_distance = _relative_distan_tigger(text, tigger_loc)
            encode_dict = tokenizer.encode_plus(text=text_,
                                                max_length=max_seq_len,
                                                pad_to_max_length=True,
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)
            token_ids = encode_dict['input_ids']
            attention_masks = encode_dict['attention_mask']
            token_type_ids = encode_dict['token_type_ids']
            ttkk = lable
            qm = len(tigger_distance)
            if max_seq_len:
                if len(lable) > max_seq_len:
                    token_ids = token_ids[:max_seq_len]
                    attention_masks = attention_masks[:max_seq_len]
                    token_type_ids = token_type_ids[:max_seq_len]
                    lable = lable[:max_seq_len]
                    tigger_distance = tigger_distance[:max_seq_len]
                else:
                    lable += pad_lables * (max_seq_len - len(ttkk))
                    tigger_distance += [0] * (max_seq_len - qm)
            examples.append((token_ids, attention_masks, token_type_ids, lable, tigger_loc, tigger_distance))
        return examples


class FGM:
    """用于对抗训练，提升鲁棒性，
    使用方式
    fgm.attack()
    loss_adv = loss_func(x, lable)
    loss_adv.backward(retain_graph=True)
    fgm.restore()
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embedding.'):
        # emb_name 这个参数要换成你模型中 embedding 的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embedding.'):
        # emb_name 这个参数要换成你模型中 embedding 的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def build_optimizer_and_scheduler(opt, model, t_total):
    module = (model.module if hasattr(model, "module") else model)
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split(".")
        if space[0] == "bert_module":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def get_base_out(model, loader, device, task_type):
    model.eval()
    with torch.no_grad():
        for idx, batch_ in enumerate(tqdm(loader, desc=f'Get {task_type} task predict logits')):
            for key in batch_.keys():
                batch_[key] = batch_[key].to(device)
            tmp_out = model(**batch_)

            yield tmp_out


def ComputeScore(start_prob, start_ids, limits=0.5, loss_=0.):
    """ compute some score......
    :return: loss_, precision, recall, f1
    """
    #
    batch_, steps_, prob_num = start_prob.shape
    _precision = (start_prob > limits).float()
    res_tmp = _precision.eq_(start_ids)
    precision_num = int(res_tmp.sum())
    precision = precision_num / (batch_ * steps_ * prob_num)

    _recall = com_re(start_ids, _precision)
    all_reacll_ids = start_ids.sum()

    recall = _recall / all_reacll_ids

    f1 = (2 * precision * recall) / (recall + precision)

    return loss_, precision, recall, f1


def com_re(start_ids, preb_):
    index_ = torch.nonzero(start_ids.reshape(-1)).squeeze()
    _preb_num = preb_.reshape(-1)[index_].sum()

    return _preb_num


def force_decode(logits, text, start_prob_base=0.5, end_prob_base=0.5):
    """
    :param logits:
    :param text:
    :param start_prob_base: 0.5
    :param end_prob_base: 0.5
    :return:
    用于强制解码，只返回符合概率的值
    """
    pass
