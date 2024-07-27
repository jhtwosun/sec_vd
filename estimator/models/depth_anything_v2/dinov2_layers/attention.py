# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PromptAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_prompt: int = 0,
        num_slice: int = 1,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)
        self.num_prompt = num_prompt
        self.num_patch = num_slice[0] * num_slice[1]
    
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        
        
        B, N, C = x.shape
        num_aux_token = 1 + self.num_prompt
        inner_seq_length = (N-num_aux_token) // self.num_patch
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cls_prompt_qkv, img_qkv = qkv[:, :num_aux_token], qkv[:, num_aux_token:]
        
        img_qkv =img_qkv.reshape(B, inner_seq_length, self.num_patch, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5) # 3, B, num_patch, num_heads, seq_length, C // num_heads
        cls_prompt_qkv = cls_prompt_qkv.reshape(B, num_aux_token, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, num_heads, num_aux_token, C // num_heads
        cls_prompt_qkv = cls_prompt_qkv.unsqueeze(2).repeat(1, 1, self.num_patch, 1, 1, 1)
        
        qkv = torch.cat([cls_prompt_qkv, img_qkv], dim=2).view(3, B*self.num_patch, self.num_heads, num_aux_token+inner_seq_length, C // self.num_heads)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        
        
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.num_patch, num_aux_token+inner_seq_length, C) # B, num_patch, num_aux_token+inner_seq_length, C
        cls_prompt_token = x[:, :num_aux_token].mean(dim=1) # TODO: mean or ...what? # B, num_aux_token, C
        img_token = x[:, num_aux_token:].reshape(B, self.num_patch, inner_seq_length, self.num_heads, C // self.num_heads).transpose(1, 2).reshape(B, N-num_aux_token, C) # B, inner_seq_length, C
        
        x = torch.cat([cls_prompt_token, img_token], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        
        
        return x

class MemEffPromptAttention(PromptAttention):
    
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)
        
        B, N, C = x.shape
        num_aux_token = 1 + self.num_prompt
        inner_seq_length = (N-num_aux_token) // self.num_patch
        
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads)#.permute(2, 0, 3, 1, 4)
        
        cls_prompt_qkv, img_qkv = qkv[:, :num_aux_token], qkv[:, num_aux_token:]
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        
        img_qkv =img_qkv.reshape(B*self.num_patch, inner_seq_length,  3, self.num_heads, C // self.num_heads)#.permute(3, 0, 1, 4, 2, 5) # 3, B, num_patch, num_heads, seq_length, C // num_heads
        cls_prompt_qkv = cls_prompt_qkv.reshape(B, num_aux_token, 3, self.num_heads, C // self.num_heads)#.permute(3, 0, 1, 4, 2, 5) # 3, B, num_heads, num_aux_token, C // num_heads
        cls_prompt_qkv = cls_prompt_qkv.repeat_interleave(self.num_patch, dim=0)
        
        #qkv = torch.cat([cls_prompt_qkv, img_qkv], dim=2)#.reshape(3, B*self.num_patch, self.num_heads, num_aux_token+inner_seq_length, C // self.num_heads)

        qkv = torch.cat([cls_prompt_qkv, img_qkv], dim=1)
        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, self.num_patch, num_aux_token+inner_seq_length, C])
        cls_prompt_token = x[:, :, :num_aux_token].mean(dim=1) # TODO: mean or ...what? # B, num_aux_token, C
        img_token = x[:, :, num_aux_token:].reshape(B, N-num_aux_token, C) # B, inner_seq_length, C
        
        x = torch.cat([cls_prompt_token, img_token], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        
        return x


class PromptAttention2(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_prompt: int = 0,
        num_slice: int = 1,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)
        self.num_prompt = num_prompt
        self.num_patch = num_slice[0] * num_slice[1]
    
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        
        
        B, N, C = x.shape
        num_aux_token = 1 + self.num_prompt
        inner_seq_length = (N-num_aux_token) // self.num_patch
        inner_prompt_length = self.num_prompt // self.num_patch
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cls_qkv, prompt_qkv, img_qkv = qkv[:, :1],qkv[:, 1:num_aux_token], qkv[:, num_aux_token:]
        
        img_qkv = img_qkv.reshape(B, inner_seq_length, self.num_patch, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5) # 3, B, num_patch, num_heads, seq_length, C // num_heads
        prompt_qkv = prompt_qkv.reshape(B, inner_prompt_length, self.num_patch, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5) # 3, B, num_patch, num_heads, seq_length, C // num_heads
        
        cls_qkv = cls_qkv.reshape(B, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, num_heads, num_aux_token, C // num_heads
        cls_qkv = cls_qkv.unsqueeze(2).repeat(1, 1, self.num_patch, 1, 1, 1)
        
        qkv = torch.cat([cls_qkv, prompt_qkv, img_qkv], dim=2).view(3, B*self.num_patch, self.num_heads, num_aux_token+inner_seq_length, C // self.num_heads)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        
        
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.num_patch, 1+inner_prompt_length+inner_seq_length, C) # B, num_patch, num_aux_token+inner_seq_length, C
        cls_token = x[:, :1].mean(dim=1) # TODO: mean or ...what? # B, num_aux_token, C
        prompt_token = x[:, 1:1+inner_prompt_length].reshape(B, self.num_patch, inner_prompt_length, self.num_heads, C // self.num_heads).transpose(1, 2).reshape(B, self.num_prompt, C) # B, inner_seq_length, C
        img_token = x[:, num_aux_token:].reshape(B, self.num_patch, inner_seq_length, self.num_heads, C // self.num_heads).transpose(1, 2).reshape(B, N-num_aux_token, C) # B, inner_seq_length, C
        
        x = torch.cat([cls_token, prompt_token, img_token], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        
        
        return x

class MemEffPromptAttention2(PromptAttention2):
    
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)
    
        B, N, C = x.shape
        num_aux_token = 1 + self.num_prompt
        inner_seq_length = (N-num_aux_token) // self.num_patch
        inner_prompt_length = self.num_prompt // self.num_patch
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)#.permute(2, 0, 3, 1, 4)
        
        cls_qkv, prompt_qkv, img_qkv = qkv[:, :1], qkv[:, 1:num_aux_token], qkv[:, num_aux_token:]
        
        
        prompt_qkv = prompt_qkv.reshape(B*self.num_patch, inner_prompt_length, 3, self.num_heads, C // self.num_heads)
        img_qkv = img_qkv.reshape(B*self.num_patch, inner_seq_length, 3, self.num_heads, C // self.num_heads)#.permute(3, 0, 1, 4, 2, 5) # 3, B, num_patch, num_heads, seq_length, C // num_heads
        #.permute(3, 0, 1, 4, 2, 5) # 3, B, num_patch, num_heads, seq_length, C // num_heads
        
        cls_qkv = cls_qkv.reshape(B, 1, 3, self.num_heads, C // self.num_heads)#.permute(3, 0, 1, 4, 2, 5) # 3, B, num_heads, num_aux_token, C // num_heads
        cls_qkv = cls_qkv.repeat_interleave(self.num_patch, dim=0)
        
        qkv = torch.cat([cls_qkv, prompt_qkv, img_qkv], dim=1)

        q, k, v = unbind(qkv, 2)      

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, self.num_patch, 1+inner_prompt_length+inner_seq_length, C])
        
        cls_token = x[:, :, :1].mean(dim=1) # TODO: mean or ...what? # B, num_aux_token, C
        prompt_token = x[:, :, 1:1+inner_prompt_length].reshape(B, self.num_prompt, C)
        img_token = x[:, :, 1+inner_prompt_length:].reshape(B, N-num_aux_token, C) # B, inner_seq_length, C
        
        x = torch.cat([cls_token, prompt_token, img_token], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        
        
        return x
