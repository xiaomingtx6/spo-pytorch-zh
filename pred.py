# -*- coding: utf-8 -*-
import numpy as np
import torch

from utils import *
from models import *

modesl = torch.load("./pre_train/pytorch_model.bin", map_location=torch.device('cpu'))

model = SOModel("./pre_train", 0.1, True)
model.load_state_dict(torch.load("./checkpoint/checkpoint-730/model.pt", map_location=torch.device('cpu')), False)
model.eval()


def _relative_distan_tigger(text, tigger_loc):
    tigger_loc_start = tigger_loc[0][1]
    tigger_loc_end = tigger_loc[0][2]
    tigger_distance_ = [0] * (len(text) + 2)
    for i in range(tigger_loc_start + 1, 0, -1):
        tigger_distance_[tigger_loc_start + 1 - i] = i
    for i in range(tigger_loc_end + 1, len(tigger_distance_)):
        tigger_distance_[i] = i + 1 - tigger_loc_end - 1
    return tigger_distance_


def _to_tensor(text):
    max_seq_len = len(text) + 2
    text = [i for i in text]
    encode_dict = tokenizer.encode_plus(text=text,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True,
                                        is_split_into_words=True,
                                        )
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']
    token_ids = torch.tensor([token_ids]).long()
    attention_masks = torch.tensor([attention_masks]).long()
    token_type_ids = torch.tensor([token_type_ids]).long()
    tiggler = [[21, 23]]

    _tigger_distance = _relative_distan_tigger(text, [["", 21, 23]])
    _tigger_distance = torch.tensor([_tigger_distance]).long()

    tigger = torch.tensor([[tiggler[0][0], tiggler[0][1]]]).long()
    tmp_out = model(token_ids, attention_masks, token_type_ids, tigger, _tigger_distance)[0]
    # print(tmp_out)
    # print(tmp_out[0])
    start_ids = np.argwhere(tmp_out[0][:, 0] > 0.5)[:, 0]
    end_ids = np.argwhere(tmp_out[0][:, 1] > 0.5)[:, 0]
    print(start_ids)
    print(end_ids)

    origin_text = ["@"] + text + ["@"]
    print("".join(origin_text)[start_ids:end_ids + 1])


if __name__ == '__main__':
    _to_tensor("2019年12月26日，长岐镇和吴阳镇连续发生两起命案")
