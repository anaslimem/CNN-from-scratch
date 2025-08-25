import numpy as np 

class Optimizer:
    def step(self, params_and_grads):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0, decoupled=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.v = {}
    
    def step(self, params_and_grads):
        for idx, (p, g) in enumerate(params_and_grads):
            if g is None:
                continue
            # Apply weight decay (L2 regularization) if enabled and not decoupled
            if not self.decoupled and self.weight_decay > 0:
                g += self.weight_decay * p 
            if self.momentum > 0:
                # Get previous velocity for this parameter (default to 0.0)
                v = self.v.get(idx, 0.0)
                # Update velocity: momentum * previous velocity - learning rate * gradient
                v = self.momentum * v - self.learning_rate * g
                self.v[idx] = v
                # Update parameter using velocity
                p += v
            else:
                # Standard SGD update (no momentum)
                p -= self.learning_rate * g
            if self.decoupled and self.weight_decay > 0.0:
                p *= (1.0 - self.learning_rate * self.weight_decay)

class Adam(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, params_and_grads):
        self.t += 1
        b1, b2 = self.betas
        for idx, (p, g) in enumerate(params_and_grads):
            if g is None:
                continue
            m = self.m.get(idx, 0.0)
            v = self.v.get(idx, 0.0)
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g**2
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            if self.weight_decay > 0.0:
                p *= (1.0 - self.lr * self.weight_decay)
            self.m[idx] = m
            self.v[idx] = v