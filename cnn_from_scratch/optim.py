import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, decoupled=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.decoupled = decoupled
        self.velocities = {}

    def step(self, params_and_grads):
        for idx, (param, grad) in enumerate(params_and_grads):
            if self.weight_decay > 0:
                if self.decoupled:
                    # Decoupled (AdamW-style)
                    param[...] -= self.lr * self.weight_decay * param
                else:
                    grad = grad + self.weight_decay * param

            if self.momentum > 0:
                if idx not in self.velocities:
                    self.velocities[idx] = np.zeros_like(grad)
                v = self.momentum * self.velocities[idx] - self.lr * grad
                self.velocities[idx] = v
                param[...] += v
            else:
                param[...] -= self.lr * grad


class AdamW:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params_and_grads):
        self.t += 1
        for idx, (param, grad) in enumerate(params_and_grads):
            if self.weight_decay > 0:
                param[...] -= self.lr * self.weight_decay * param

            if idx not in self.m:
                self.m[idx] = np.zeros_like(grad)
                self.v[idx] = np.zeros_like(grad)

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            param[...] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
