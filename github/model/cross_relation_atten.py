import copy

import torch.nn as nn
import math
import torch
import torch.nn.functional as F

def randbool(size, p=0.85):
    return torch.rand(*size) < p

def attention(q, k, v, d_k, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

def masked_attention(q, k, v, d_k, mask=True, dropout=None):
    scores1 = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores1, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)

    masked_scores = scores1

    if mask is True:
        maskfill = randbool(masked_scores.shape).to(masked_scores.device)
        masked_scores = masked_scores.masked_fill(maskfill, 0.0)

    masked_scores = F.softmax(masked_scores, dim=-1)
    if dropout is not None:
        masked_scores = dropout(masked_scores)
    masked_output = torch.matmul(masked_scores, v)

    return output, masked_output

class MaskRelationMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, q_d_model, kv_d_model, dropout=0.5):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(q_d_model, d_model)
        self.v_linear = nn.Linear(kv_d_model, d_model)
        self.k_linear = nn.Linear(kv_d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, q_d_model)

    def forward(self, q, k, v, mask=False):
        bs = q.size(0)

        # perform linear operation and split into N heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores, masked_scores = masked_attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, self.d_model)
        masked_concat = masked_scores.transpose(1, 2).contiguous().view(bs, self.d_model)

        output = self.out(concat)
        masked_output = self.out(masked_concat)

        return output, masked_output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, q_d_model, kv_d_model, dropout=0.5):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(q_d_model, d_model)
        self.v_linear = nn.Linear(kv_d_model, d_model)
        self.k_linear = nn.Linear(kv_d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, q_d_model)

    def forward(self, q, k, v):
        bs = q.size(0)

        # perform linear operation and split into N heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, self.d_model)
        output = self.out(concat)

        return output


class CrossAttention(nn.Module):
    def __init__(self,heads,d_model,tumor_dim,node_dim):
        super(CrossAttention, self).__init__()
        self.tumor_MHA = MultiHeadAttention(heads=heads, d_model=d_model, q_d_model=tumor_dim, kv_d_model=tumor_dim)
        self.node_MHA = MultiHeadAttention(heads=heads, d_model=d_model, q_d_model=node_dim, kv_d_model=node_dim)

        self.tumor_MaskCrossMHA = MaskRelationMultiHeadAttention(heads=heads, d_model=d_model, q_d_model=tumor_dim, kv_d_model=tumor_dim)
        self.node_MaskCrossMHA = MaskRelationMultiHeadAttention(heads=heads, d_model=d_model, q_d_model=node_dim, kv_d_model=node_dim)

    def forward(self,tumor_embed,node_embed):
        tumor_embed_mha = self.tumor_MHA(tumor_embed,tumor_embed,tumor_embed)
        node_embed_mha = self.node_MHA(node_embed,node_embed,node_embed)
        #
        tumor_output_unmasked_cross_mha, tumor_output_masked_cross_mha = self.tumor_MaskCrossMHA(tumor_embed,node_embed, node_embed )
        node_output_unmasked_cross_mha, node_output_masked_cross_mha = self.node_MaskCrossMHA(node_embed, tumor_embed, tumor_embed)

        # print(tumor_output_unmasked_cross_mha.shape)
        # print(tumor_output_masked_cross_mha.shape)
        # print(node_output_unmasked_cross_mha.shape)
        # print(node_output_masked_cross_mha.shape)
        # return  tumor_embed_mha, node_embed_mha

        return tumor_output_unmasked_cross_mha, tumor_output_masked_cross_mha, node_output_unmasked_cross_mha, node_output_masked_cross_mha, \
               tumor_embed_mha, node_embed_mha

if __name__ == '__main__':
    tumor = torch.rand((2,128))
    node = torch.rand((2,128))
    model = CrossAttention(heads=4,d_model=128,tumor_dim=128,node_dim=128)
    k,k1,k2,k3 = model(tumor,node)
    print(k.shape)
    print(k1.shape)