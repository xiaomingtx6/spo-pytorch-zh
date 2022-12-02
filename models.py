# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from transformers import BertModel


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))

        self.weight_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)

        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        #print(cond)
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs * weight + bias

        return outputs


class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir)

        print(self.bert_module.state_dict().keys())
        #print(self.bert_module.state_dict().keys())
        self.bert_config = self.bert_module.config
        self.dropout_layer = nn.Dropout(dropout_prob)

    def _init_weights(self, blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)

    def _batch_gather(self, data: torch.Tensor, index: torch.Tensor):
        """
        实现类似 tf.batch_gather 的效果
        :param data: (bs, max_seq_len, hidden)
        :param index: (bs, n)
        :return: a tensor which shape is (bs, n, hidden)
        """
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)
        return torch.gather(data, 1, index)


class SOModel(BaseModel):
    def __init__(self, bert_dir, dropout_prob=0.1, use_tigger_distance=False, **kwargs):
        super(SOModel, self).__init__(bert_dir, dropout_prob)
        out_dims = self.bert_config.hidden_size
        self.conditional_layer_norm = ConditionalLayerNorm(out_dims, eps=self.bert_config.layer_norm_eps)  #
        self.use_trigger_distance = use_tigger_distance
        if use_tigger_distance:
            embedding_dim = kwargs.pop('embedding_dims', 256)

            self.trigger_distance_embedding = nn.Embedding(num_embeddings=512, embedding_dim=embedding_dim)

            out_dims += embedding_dim

            self.layer_norm = nn.LayerNorm(out_dims, eps=self.bert_config.layer_norm_eps)
        mid_linear_dims = kwargs.pop("mid_linear_dims", 128)

        self.mid_linear = nn.Sequential(nn.Linear(out_dims, mid_linear_dims),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_prob))

        self.obj_classifier = nn.Linear(mid_linear_dims, 2)
        self.sub_classifier = nn.Linear(mid_linear_dims, 2)

        self.activation = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        init_blocks = [self.mid_linear, self.obj_classifier, self.sub_classifier]

        if use_tigger_distance:
            init_blocks += [self.trigger_distance_embedding, self.layer_norm]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                trigger_index,
                trigger_distance=None,
                labels=None):
        bert_outputs = self.bert_module(input_ids=token_ids, attention_mask=attention_masks,
                                        token_type_ids=token_type_ids)


        seq_out, pooled_out = bert_outputs[0], bert_outputs[1]
        #seq_out, _,_,_ = bert_outputs

        tigger_lable_features = self._batch_gather(seq_out, trigger_index)
        tigger_lable_features = tigger_lable_features.view([tigger_lable_features.size()[0], -1])
        seq_out = self.conditional_layer_norm(seq_out, tigger_lable_features)
        if self.use_trigger_distance:
            assert trigger_distance is not None, \
                'When using trigger distance features, trigger distance should be implemented'
            trigger_distance_feature = self.trigger_distance_embedding(trigger_distance)
            seq_out = torch.cat([seq_out, trigger_distance_feature], dim=-1)
            seq_out = self.layer_norm(seq_out)

        seq_out = self.mid_linear(seq_out)

        obj_logits = self.activation(self.obj_classifier(seq_out))
        sub_logits = self.activation(self.sub_classifier(seq_out))

        logits = torch.cat([obj_logits, sub_logits], dim=-1)
        out = (logits,)
        if labels is not None:
            masks = torch.unsqueeze(attention_masks, -1)

            labels = labels.float()
            obj_loss = self.criterion(obj_logits * masks, labels[:, :, :2])
            sub_loss = self.criterion(sub_logits * masks, labels[:, :, 2:])

            loss = obj_loss + sub_loss
            out = (loss,) + out

        return out


class TiggerCRF():
    def __init__(self):
        """对触发词对提取采用bert+crf模式
        """
        super(TiggerCRF, self).__init__()
