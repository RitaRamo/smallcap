# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.gpt2.modeling_gpt2 import load_tf_weights_in_gpt2, GPT2LMHeadModel, GPT2MLP, GPT2Attention, GPT2Block, GPT2Model 

from transformers.models.xglm.modeling_xglm import XGLMForCausalLM, XGLMAttention, XGLMDecoderLayer, XGLMModel 

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.xglm.configuration_xglm import XGLMConfig


if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False

    
class ThisXGLMConfig(XGLMConfig):
    model_type = "this_xglm"

    def __init__(
        self,
        cross_attention_reduce_factor = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor
        

class ThisXGLMAttention(XGLMAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout= 0.0,
        is_decoder= False,
        bias= True,
        config=None,
        is_cross_attention=False,
    ):
        super().__init__(embed_dim,num_heads, dropout,is_decoder,bias)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        self.head_dim = int(self.head_dim / self.cross_attention_reduce_factor)


        if is_cross_attention:
            #print("self", int(embed_dim / self.cross_attention_reduce_factor))
            self.k_proj = nn.Linear(768, int(embed_dim / self.cross_attention_reduce_factor), bias=bias)
            #print("self.k_proj",self.k_proj)
            self.v_proj = nn.Linear(768, int(embed_dim / self.cross_attention_reduce_factor), bias=bias)
            self.q_proj = nn.Linear(embed_dim, int(embed_dim / self.cross_attention_reduce_factor), bias=bias)
            self.out_proj = nn.Linear(int(embed_dim / self.cross_attention_reduce_factor),embed_dim, bias=bias)

            self.embed_dim=int(embed_dim / self.cross_attention_reduce_factor)
        else:
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim , bias=bias)
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states,
        key_value_states,
        past_key_value,
        attention_mask,
        layer_head_mask,
        output_attentions,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            #print("key_value_states",key_value_states.size())
            #print("self.k_proj(key_value_states)",self.k_proj(key_value_states).size())
                
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        #print("boraaa self.head_dim",self.head_dim)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        
        #print("attn_output bef",attn_output.size())
        attn_output = attn_output.transpose(1, 2)
        #print("attn_output",attn_output.size())
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        #print("attn_output",attn_output.size())
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class ThisXGLMDecoderLayer(XGLMDecoderLayer):
    def __init__(self, config):
        super().__init__(config)

        if config.add_cross_attention:
            print("add cross")
            self.encoder_attn = ThisXGLMAttention(
                embed_dim=self.embed_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                config=config,
                is_cross_attention=True
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

class ThisXGLMModel(XGLMModel):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([ThisXGLMDecoderLayer(config) for _ in range(config.num_layers)])

class ThisXGLMForCausalLM(XGLMForCausalLM):
    config_class = ThisXGLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = ThisXGLMModel(config)

