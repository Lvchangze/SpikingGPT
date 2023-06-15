import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

# from spikingjelly.activation_based import neuron, functional, layer, surrogate
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode

from torch.utils.data import Dataset
import math
import numpy as np
from torch.nn import functional as F


tau = 1.2
v_threshold = 1.
# surrogate = surrogate.ATan()
backend = 'torch'
laynorm_eps = 1e-12


'''
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                        输入是经过tokenizer后的---> B * L, 然后repeat成 T * B * L, (L=64)
                                                        但这里，输入先是 B * L,  在经过SPS后repeat带上时间步
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
class GPTConfig(object):
    def __init__(self, 
                 time_step=4,
                 hidden_dim=512,
                 vocab_size=3000,
                 block_size=128,
                 num_heads=8,
                 depths=2,
                 ):
        self.time_step = time_step
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_heads = num_heads
        self.depths = depths
        pass


# config = GPTConfig()





class SPS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_step = config.time_step
        self.emb = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.fc = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ln = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        # self.lif = nn.GELU()

    def forward(self, x):
        B, L = x.shape
        x = self.emb(x)
        x = self.ln(self.fc(x))
        x = x.unsqueeze(0).repeat(self.time_step, 1, 1, 1)
        x = self.lif(x)
        return x  # T B L D


# print('---------------------  SPS  ---------------------')
# x = torch.randint(low=0, high=1000, size=(16, 64))
# model = SPS(config)
# y = model(x)
# print(y.shape)
# print("最小值:", torch.min(y))
# print("最大值:", torch.max(y))
# # print("有这些值：", torch.unique(y))
# print("元素的值类型个数", len(torch.unique(y)))


class SSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0, f"dim {config.hidden_dim} should be divided by num_heads {config.num_heads}."
        self.dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.scale = 0.125

        self.qkv_fc = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.ln = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)

        self.q_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
        self.k_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
        self.v_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        self.attn_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend=backend)
        self.fc = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)
        
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, 1, config.block_size, config.block_size))

    def forward(self, x):
        T, B, L, D = x.shape
        qkv = self.qkv_fc(x.flatten(0, 1)).reshape(T * B, L, 3, D).permute(2, 0, 1, 3)  # qkv = [3, TB, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每一项： [TB, L, D]

        q = self.q_lif(self.ln(q).reshape(T, B, L, D))
        k = self.k_lif(self.ln(k).reshape(T, B, L, D))
        v = self.v_lif(self.ln(v).reshape(T, B, L, D))  # T B L D

        q = q.reshape(T, B, L, self.num_heads, D//self.num_heads).permute(0, 1, 3, 2, 4)
        k = k.reshape(T, B, L, self.num_heads, D//self.num_heads).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, L, self.num_heads, D//self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.mask[:, :, :, :L, :L] == 0, 0)

        # Spikformer
        x = attn @ v
        x = x.transpose(2, 3).flatten(-2, -1)
        x = self.attn_lif(x).flatten(0, 1)
        x = self.ln2(self.fc(x)).reshape(T, B, L, D)

        # Spikingformer
        x = attn @ v
        x = x.transpose(2, 3).flatten(-2, -1)
        x = self.ln2(self.fc(x)).reshape(T, B, L, D)
        x = self.attn_lif(x)

        return x





# print('---------------------  SSA  ---------------------')
# model = SSA(config)
# y = model(y)
# print(y.shape)
# print("最小值:", torch.min(y))
# print("最大值:", torch.max(y))
# # print("有这些值：", torch.unique(y))
# print("元素的值类型个数", len(torch.unique(y)))
# # print(y[0][0])


# Spikingformer
# class MLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
#         self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
#         self.ln1 = nn.LayerNorm(config.hidden_dim * 4, eps=laynorm_eps)

#         self.lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
#         self.fc2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
#         self.ln2 = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)

#     def forward(self, x):
#         T, B, L, D = x.shape
#         x = self.lif1(x)
#         x = self.ln1(self.fc1(x.flatten(0, 1))).reshape(T, B, L, 4*D)

#         x = self.lif2(x)
#         x = self.ln2(self.fc2(x.flatten(0, 1))).reshape(T, B, L, D)

#         return x


# Spik
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.ln1 = nn.LayerNorm(config.hidden_dim * 4, eps=laynorm_eps)
        self.lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        self.fc2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)
        self.lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

    def forward(self, x):
        T, B, L, D = x.shape
        x = self.lif1(self.ln1(self.fc1(x)))
        x = self.lif2(self.ln2(self.fc2(x)))

        return x





# print('---------------------  MLP  ---------------------')
# model = MLP(config)
# y = model(y)
# print(y.shape)

# print("最小值:", torch.min(y))
# print("最大值:", torch.max(y))
# # print("有这些值：", torch.unique(y))
# print("元素的值类型个数", len(torch.unique(y)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = SSA(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# print('---------------------  Block  ---------------------')
# model = Block(config)
# y = model(y)
# print(y.shape)
# print("最小值:", torch.min(y))
# print("最大值:", torch.max(y))
# # print("有这些值：", torch.unique(y))
# print("元素的值类型个数", len(torch.unique(y)))




class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.emb = SPS(config)
        self.block = nn.ModuleList([Block(config) for j in range(config.depths)])

        # classification head
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def get_block_size(self):
        return self.block_size

    def forward(self, x,  targets=None):
        x = self.emb(x)
        state = []
        for blk in self.block:
            x = blk(x)
            state.append(x)

        logits = self.head(state[-1]).mean(0)  # B * L * vocab_size

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss



# print('---------------------  Spikformer  ---------------------')
# x = torch.randint(low=0, high=1000, size=(16, 64))
# target = torch.randint(0, 3000, [16, 64])
# model = GPT(config)
# # state, y = model(x)
# y, loss = model(x, target)
# print(f"loss = {loss}")
# print(y.shape)
# print("最小值:", torch.min(y))
# print("最大值:", torch.max(y))
# # print("有这些值：", torch.unique(y))
# print("元素的值类型个数", len(torch.unique(y)))
