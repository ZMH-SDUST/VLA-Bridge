# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/29 9:50
@Auther ： Zzou
@File ：HOI_Decoder.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import torch
from detr.models.transformer import *

d_model = 512
nhead = 4
dim_feedforward = 512
dropout = 0.1
normalize_before = False
num_decoder_layers = 2
return_intermediate_dec = True

decoder_norm = nn.LayerNorm(d_model)
# decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
#                                         dropout, activation, normalize_before)
# hoi_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
#                                  return_intermediate=return_intermediate_dec)
self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
Linear = nn.Linear(d_model * 2, d_model)
norm_q = nn.LayerNorm(d_model)
norm_k = nn.LayerNorm(d_model)
norm_v = nn.LayerNorm(d_model)
norm_hl2 = nn.LayerNorm(d_model)
Linear_2 = nn.Linear(d_model, d_model)
Dropout = nn.Dropout(dropout)
activation = F.relu

num_pairs = 27
feat_global = torch.zeros([1, 512])  # CLIP global features I
local_features = torch.zeros([512, 14, 14])  # CLIP patch features
human_features = torch.zeros([num_pairs, 512])  # human instance CLIP features
object_features = torch.zeros([num_pairs, 512])  # object instance CLIP features
human_object_features = torch.cat([human_features, object_features],
                                  dim=-1)  # concat human + object instances CLIP features [num_pairs, 512]
union_features = torch.zeros([num_pairs, 512])  # union pairs instance CLIP features


feat_global = feat_global.unsqueeze(0)
H0 = Linear(human_object_features).unsqueeze(1)  # q:[27, 1, 512]
F = local_features.reshape(local_features.shape[0], -1).transpose(0, 1).unsqueeze(1)  # k,v:[14*14, 1, 512]
Hl1 = cross_attn(query=norm_q(H0), key=norm_k(F), value=norm_v(F))[0]
Hl1 = Hl1 + H0
qkv_Hl2 = torch.cat([feat_global, Hl1], 0) # [28, 1, 512]
Hl2 = self_attn(query=norm_hl2(qkv_Hl2), key=norm_hl2(qkv_Hl2), value=norm_hl2(qkv_Hl2))[0] # [28, 1, 512]
Hl2 = Hl2 + qkv_Hl2
Hl = activation(Linear_2(Hl2[1:])) # 27, 1, 512
P = Hl
