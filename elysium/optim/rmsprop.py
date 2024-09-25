from elysium import cp,np
from elysium import zeros_like,no_grad
from .optimizer import Optim
class RMSprop(Optim):
    def __init__(self, model, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super().__init__(model)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        # Initialize square average and momentum buffers
        self.square_avg = {param_name:zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters}
        self.momentum_buffers = {param_name:zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters if momentum > 0}
        if self.centered:
            self.grad_avg = {param_name:zeros_like(param_value,device=param_value.device) for param_name, param_value in self.parameters}

    def step(self):
        for param_name, param_value in self.parameters:
            grad = param_value.grad.data
            if grad is None:continue  # Skip if no gradient is available
            # Apply weight decay if specified
            if self.weight_decay != 0:grad += self.weight_decay * param_value.data
            self.square_avg[param_name].data *= self.alpha
            self.square_avg[param_name] += (1 - self.alpha) * (grad ** 2)
            if self.centered:
                self.grad_avg[param_name].data *= self.alpha
                self.grad_avg[param_name].data += (1 - self.alpha) * grad
                avg = self.square_avg[param_name].data - self.grad_avg[param_name].data ** 2
            else:
                avg = self.square_avg[param_name].data

            if self.momentum > 0:
                self.momentum_buffers[param_name].data *= self.momentum
                self.momentum_buffers[param_name].data += grad / ((cp if (cp is not None and param_value.data.__class__ is cp.ndarray ) else np).sqrt(avg) + self.eps)
                param_value.data -= self.lr * self.momentum_buffers[param_name].data
            else:
                param_value -= self.lr * grad / (((cp if (cp is not None and param_value.data.__class__ is cp.ndarray ) else np)).sqrt(avg) + self.eps)
