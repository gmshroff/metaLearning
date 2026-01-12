import numpy as np
from autograd import Tensor
from nn import Module, Linear, LayerNorm, Embedding, ReLU

class CausalSelfAttention(Module):
    def __init__(self, n_embd, n_head, block_size):
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.c_attn = Linear(n_embd, 3 * n_embd)
        self.c_proj = Linear(n_embd, n_embd)
        
        # Causal mask
        self.register_buffer("bias", np.tril(np.ones((block_size, block_size)))
                                     .reshape(1, 1, block_size, block_size))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def __call__(self, x):
        B, T, C = x.data.shape 
        
        # qkv: (B, T, 3 * C)
        qkv = self.c_attn(x)
        
        # Reshape to (B, T, 3, n_head, head_dim)
        qkv = qkv.reshape((B, T, 3, self.n_head, self.head_dim))
        
        # Transpose to (3, B, n_head, T, head_dim)
        qkv = qkv.transpose((2, 0, 3, 1, 4))
        
        # Split q, k, v
        q = qkv[0] # (B, n_head, T, head_dim)
        k = qkv[1]
        v = qkv[2]
        
        # Attention scores: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        # We need to transpose k to (B, n_head, head_dim, T)
        k_t = k.transpose((0, 1, 3, 2))
        
        att = (q @ k_t) * (1.0 / np.sqrt(self.head_dim))
        
        # Masking
        # We need to apply mask. Since we don't have masked_fill, we can use addition.
        # bias is (1, 1, block_size, block_size). We slice it to (1, 1, T, T)
        mask = self.bias[:, :, :T, :T]
        # We want to set positions where mask == 0 to -inf.
        # att = att + (mask == 0) * -1e9
        # But mask is numpy array. We need to make it a Tensor or just use data if we don't backprop through mask (we don't).
        # Actually, we can just do:
        # att.data = np.where(mask == 1, att.data, -1e9)
        # But modifying .data in place breaks the graph if we are not careful? No, it's fine for forward, but backward?
        # If we modify data, the backward pass of previous ops relies on 'out.grad'.
        # The 'att' tensor is new.
        # But we need to make sure the operation is recorded.
        # If we just modify data, it's not recorded.
        # We should use an op.
        # att = att + (mask - 1) * 1e9 ?
        # If mask is 1, (1-1)*1e9 = 0. If mask is 0, (0-1)*1e9 = -1e9.
        # Yes.
        
        att = att + Tensor((mask - 1) * 1e9)
        
        # Softmax
        # We need softmax along the last dimension.
        # My Softmax in nn.py takes dim.
        # But here we are doing it manually or using nn.Softmax?
        # Let's use a local softmax implementation or update nn.Softmax to be reusable.
        # nn.Softmax(dim=-1) should work.
        
        # att = att.softmax(dim=-1) # Not implemented in Tensor
        # Let's use the logic from nn.Softmax
        att_max = att.data.max(axis=-1, keepdims=True)
        att = att - Tensor(att_max)
        att_exp = att.exp()
        att_sum = att_exp.sum(axis=-1, keepdims=True)
        att = att_exp / att_sum
        
        # Weighted sum of values
        y = att @ v # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        
        # Reassemble
        y = y.transpose((0, 2, 1, 3)) # (B, T, n_head, head_dim)
        y = y.reshape((B, T, C))
        
        return self.c_proj(y)

    def parameters(self):
        return self.c_attn.parameters() + self.c_proj.parameters()

class Block(Module):
    def __init__(self, n_embd, n_head, block_size):
        self.ln1 = LayerNorm((n_embd,))
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = LayerNorm((n_embd,))
        self.mlp_l1 = Linear(n_embd, 4 * n_embd)
        self.relu = ReLU()
        self.mlp_l2 = Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        
        # MLP
        m = self.mlp_l1(self.ln2(x))
        m = self.relu(m)
        m = self.mlp_l2(m)
        
        x = x + m
        return x
        
    def parameters(self):
        return self.ln1.parameters() + self.attn.parameters() + self.ln2.parameters() + \
               self.mlp_l1.parameters() + self.mlp_l2.parameters()

class GPT(Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        self.token_embedding = Embedding(vocab_size, n_embd)
        self.position_embedding = Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        self.ln_f = LayerNorm((n_embd,))
        self.lm_head = Linear(n_embd, vocab_size)
        self.block_size = block_size

    def __call__(self, idx):
        B, T = idx.data.shape
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(Tensor(np.arange(T)))
        
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
        
    def parameters(self):
        params = self.token_embedding.parameters() + self.position_embedding.parameters() + \
                 self.ln_f.parameters() + self.lm_head.parameters()
        for block in self.blocks:
            params += block.parameters()
        return params
