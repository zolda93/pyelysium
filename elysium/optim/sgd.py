from elysium import zeros_like,no_grad
from .optimizer import Optim
class SGD(Optim):
    def __init__(self, model, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False):
        super().__init__(model)
        if lr<0.0:raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:raise ValueError("Invalid momntum value: {}".format(momentum))
        if weight_decay < 0.0:raise ValueError("Invalid weight_decay value: {}".fomat(weight_decay))
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self.momentum_buffers = {param_name:zeros_like(param_value,device=param_value.device)
                                 for param_name, param_value in self.parameters if momentum > 0}
    def step(self):
        for param_name,param_value in self.parameters:
            grad = param_value.grad.data
            if grad is None:continue
            if self.weight_decay!=0:
                grad += self.weight_decay*param_value.data
            if self.momentum!=0:
                self.momentum_buffers[param_name].data = self.momentum*self.momentum_buffers[param_name].data + (1 - self.dampening) * grad
                if self.nesterov:
                    grad += self.momentum * self.momentum_buffers[param_name].data
                else:
                    grad= self.momentum_buffers[param_name]
            if self.maximize:
                param_value.data += self.lr * grad
            else:
                param_value.data -= self.lr*grad




