from elysium import cp,np
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
        self.t = 0
        # Initialize state variables for first and second moment estimates
        self.exp_avg = {param_name: zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters}
        self.exp_avg_seq = {param_name: zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters}
        if amsgrad:
            self.max_exp_avg_seq = {param_name:zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters}
    def step(self):
        self.t += 1
        for param_name, param in self.parameters:
            grad = param.grad.data
            if grad is None:continue  # Skip if no gradient is available
            param.data *= 1 - self.lr * self.weight_decay
            # Update first and second moment estimates
            bias_correction1 = 1 - self.betas[0] ** self.t
            bias_correction2 = 1 - self.betas[1] ** self.t

            self.exp_avg[param_name].data *= self.betas[0]
            self.exp_avg[param_name].data += (1 - self.betas[0]) * grad
            self.exp_avg_seq[param_name].data *= self.betas[1]
            self.exp_avg_seq[param_name].data +=  (1 - self.betas[1]) * grad * grad.conj()
            #m_hat = self.m[param_name].data / (1 - self.betas[0]**self._step)
            #v_hat = self.v[param_name].data
            xp = cp if (cp is not None and param.data.__class__ is cp.ndarray ) else np
            if self.amsgrad:
                xp.maximum(self.max_exp_avg_seq[param_name].data, self.exp_avg_seq[param_name].data,out=self.max_exp_avg_seq[param_name].data)
                denom = (xp.sqrt(self.max_exp_avg_seq[param_name]) / xp.sqrt(bias_correction2)) + self.eps
                #param_value.data -= self.lr* m_hat / (xp.sqrt(self.v_hat[param_name].data / (1 - self.betas[1]**self._step)) + self.eps)
            else:
                denom = (xp.sqrt(self.exp_avg_seq[param_name].data) / xp.sqrt(bias_correction2)) + self.eps
                #param_value.data -= self.lr * m_hat / ((cp if (cp is not None and param_value.data.__class__ is cp.ndarray ) else np).sqrt(v_hat / (1 - self.betas[1]**self._step)) + self.eps)
            step_size = self.lr / bias_correction1
            param.data -= step_size * self.exp_avg[param_name].data / denom
