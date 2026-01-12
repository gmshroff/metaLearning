import numpy as np
from autograd import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, nin, nout):
        # Xavier initialization
        self.w = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / (nin + nout)))
        self.b = Tensor(np.zeros((1, nout)))

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]

class ReLU(Module):
    def __call__(self, x):
        return x.relu()

class MSELoss(Module):
    def __call__(self, y_pred, y_true):
        # y_true might be a numpy array or a Tensor
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        
        diff = y_pred - y_true
        return (diff**2).sum() / Tensor(float(np.prod(y_pred.data.shape)))

        return (diff**2).sum() / Tensor(float(np.prod(y_pred.data.shape)))

class Softmax(Module):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, x):
        x_max = x.data.max(axis=self.dim, keepdims=True)
        shifted_x = x - Tensor(x_max)
        exps = shifted_x.exp()
        sum_exps = exps.sum(axis=self.dim, keepdims=True)
        return exps / sum_exps

class LogSoftmax(Module):
    def __init__(self, dim=1):
        self.dim = dim
        
    def __call__(self, x):
        x_max = x.data.max(axis=self.dim, keepdims=True)
        shifted_x = x - Tensor(x_max)
        exps = shifted_x.exp()
        sum_exps = exps.sum(axis=self.dim, keepdims=True)
        return shifted_x - sum_exps.log()

class NLLLoss(Module):
    def __call__(self, input, target):
        # input: (batch_size, num_classes) - log probabilities
        # target: (batch_size,) - class indices
        
        # We need to select the values corresponding to the target classes.
        # This requires advanced indexing which our Tensor class doesn't support.
        # We can implement it using one-hot encoding and multiplication.
        
        batch_size = input.data.shape[0]
        # Create one-hot encoding of target
        # target is numpy array or Tensor
        if isinstance(target, Tensor):
            target = target.data
            
        one_hot = np.zeros_like(input.data)
        one_hot[np.arange(batch_size), target.astype(int).flatten()] = 1
        
        # Select the log probs
        selected_log_probs = input * Tensor(one_hot)
        
        # Sum and negate
        loss = -selected_log_probs.sum() / Tensor(float(batch_size))
        return loss

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim))
        
    def __call__(self, idx):
        # idx: (batch_size, seq_len) of integers
        # We need to gather embeddings.
        # Since our Tensor doesn't support advanced indexing in the graph,
        # we can use one-hot encoding multiplication if we want to be pure,
        # or we can implement a gather operation in autograd.
        # Given the complexity, let's implement a simple forward pass that creates a new Tensor
        # and manually sets the backward pass to do scatter add.
        
        # Actually, let's do the manual backward pass approach here for simplicity in this file,
        # or better, add a 'gather' or 'embedding' op to Tensor?
        # No, let's keep it in nn.py if possible.
        
        # Let's try the one-hot approach. It's memory intensive but clean with current ops.
        # Wait, one-hot for vocab size 65 (shakespeare) is fine.
        # batch * seq_len * vocab_size * embed_dim might be big but manageable.
        # Actually, let's implement a custom backward for Embedding here.
        
        out_data = self.weight.data[idx.data.astype(int)]
        out = Tensor(out_data, (self.weight,), 'embedding')
        
        def _backward():
            # We need to scatter add out.grad into self.weight.grad
            # np.add.at is useful here.
            # d_weight = np.zeros_like(self.weight.data)
            # np.add.at(d_weight, idx.data, out.grad)
            # self.weight.grad += d_weight
            
            # Note: idx is not differentiable, so no grad for it.
            
            # We need to flatten idx and out.grad to use add.at easily
            grad = out.grad.reshape((-1, out.grad.shape[-1]))
            indices = idx.data.reshape(-1).astype(int)
            np.add.at(self.weight.grad, indices, grad)
            
        out._backward = _backward
        return out
        
    def parameters(self):
        return [self.weight]

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = Tensor(np.ones(normalized_shape))
        self.beta = Tensor(np.zeros(normalized_shape))
        self.eps = eps
        
    def __call__(self, x):
        # x: (batch, seq, embed)
        # mean and var over last dim
        mean = x.sum(axis=-1, keepdims=True) / Tensor(float(x.data.shape[-1]))
        # var = ((x - mean)**2).sum(axis=-1, keepdims=True) / Tensor(x.data.shape[-1]) 
        # Note: simplistic var, usually use x^2 mean - mean^2
        
        x_shift = x - mean
        var = (x_shift**2).sum(axis=-1, keepdims=True) / Tensor(float(x.data.shape[-1]))
        
        std = (var + self.eps)**0.5
        y = x_shift / std
        return y * self.gamma + self.beta
        
    def parameters(self):
        return [self.gamma, self.beta]

class AdamW:
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0
        
    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None: continue
            
            # Weight decay
            p.data -= self.lr * self.weight_decay * p.data
            
            # Adam
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (p.grad**2)
            
            m_hat = self.m[i] / (1 - self.betas[0]**self.t)
            v_hat = self.v[i] / (1 - self.betas[1]**self.t)
            
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
