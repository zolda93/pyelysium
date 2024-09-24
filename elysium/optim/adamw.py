import elysium as e
from elysium import zeros_like,no_grad
from .optimizer import Optim

class AdamW(Optim):
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False):
        super().__init__(model)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self._step = 0
        # Initialize state variables for first and second moment estimates
        self.m = {param_name: zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters}
        self.v = {param_name: zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters}
        if amsgrad:
            self.v_hat = {param_name:zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters}
    def step(self):
        with e.no_grad():
            self._step += 1
            for param_name, param_value in self.parameters:
                grad = param_value.grad
                #if grad is None:continue  # Skip if no gradient is available
                param_value *= (1 - self.lr * self.weight_decay )
                # Update first and second moment estimates
                self.m[param_name] = self.betas[0] * self.m[param_name] + (1 - self.betas[0]) * grad
                self.v[param_name] = self.betas[1] * self.v[param_name] + (1 - self.betas[1]) * (grad ** 2)
                m_hat = self.m[param_name] / (1 - self.betas[0]**self._step)
                v_hat = self.v[param_name] / (1 - self.betas[1]**self._step)
                if self.amsgrad:
                    e.Tensor.maximum(self.v_hat[param_name], v_hat,out=self.v_hat[param_name])
                    param_value -= self.lr* m_hat / (self.v_hat[param_name].sqrt() + self.eps)
                else:
                    param_value -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
