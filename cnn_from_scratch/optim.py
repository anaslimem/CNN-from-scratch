
import numpy as np

class Optimizer:
    def step(self, params_and_grads):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0, decoupled=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.v = {}

    def step(self, params_and_grads):
        for idx, (p, g) in enumerate(params_and_grads):
            if g is None: 
                continue
            if not self.decoupled and self.weight_decay > 0.0:
                g = g + self.weight_decay * p  # coupled L2
            if self.momentum > 0.0:
                v = self.v.get(idx, 0.0)
                v = self.momentum * v - self.lr * g
                self.v[idx] = v
                p += v
            else:
                p -= self.lr * g
            if self.decoupled and self.weight_decay > 0.0:
                p *= (1.0 - self.lr * self.weight_decay)

class AdamW(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params_and_grads):
        self.t += 1
        b1, b2 = self.betas
        for idx, (p, g) in enumerate(params_and_grads):
            if g is None: 
                continue
            m = self.m.get(idx, 0.0)
            v = self.v.get(idx, 0.0)
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*(g*g)
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            if self.weight_decay > 0.0:
                p *= (1.0 - self.lr * self.weight_decay)
            self.m[idx] = m
            self.v[idx] = v
