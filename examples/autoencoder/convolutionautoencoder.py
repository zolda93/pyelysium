import elysium
from elysium import nn
from elysium import optim
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,Normalize,ToTensor,Lambda
from torch_snippets.torch_loader import Report
image_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5],std=[0.5]),
    ])
training_data = MNIST('./mnist/',transform=image_transform,train=True,download=False)
testing_data = MNIST('./mnist/',transform=image_transform,train=False,download=False)
class ConvAutoEncoder:
    def __init__(self):
        self.encoder = nn.Sequential(
                nn.Conv2d(1,32,3,stride=3,padding=1,device='gpu'),
                nn.ReLU(),
                nn.AvgPool2d(2,stride=2),
                nn.Conv2d(32,64,3,stride=2,padding=1,device='gpu'),
                nn.ReLU(),
                nn.AvgPool2d(2,stride=1),)
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64,32,3,stride=2,device='gpu'),
                nn.ReLU(),
                nn.ConvTranspose2d(32,16,5,stride=3,padding=1,device='gpu'),
                nn.ReLU(),
                nn.ConvTranspose2d(16,1,2,stride=2,padding=1,device='gpu'),
                nn.Tanh(),)
        self.loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self,lr=0.001,weight_decay=1e-5)
        self.train = DataLoader(training_data,batch_size=64,shuffle=True)
        self.test = DataLoader(testing_data,batch_size=64,shuffle=True)
    def __call__(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def train_batch(self,x):
        self.optimizer.zero_grad()
        output = self(x)
        loss = self.loss(output,x)
        loss.backward()
        self.optimizer.step()
        return loss
if __name__ == '__main__':
    model = ConvAutoEncoder()
    num_epochs = 7
    log = Report(num_epochs)
    for epoch in range(num_epochs):
        N = len(model.train)
        for ix, (data, _) in enumerate(model.train):
            data = elysium.tensor(data.numpy()).cuda()
            loss = model.train_batch(data)
            log.record(epoch+1,trn_loss=loss.item(),end='\r')
        log.report_avgs(epoch+1)
    import numpy as np
    import matplotlib.pyplot as plt
    from torch_snippets import *
    for _ in range(5):
        idx = np.random.randint(len(testing_data))
        im,_= testing_data[idx]
        im = elysium.tensor(im.unsqueeze(0).numpy()).to('gpu')
        _im = model(im)
        fig, ax = plt.subplots(1,2,figsize=(3,3))
        show(im.numpy().squeeze(), ax=ax[0], title='input')
        show(_im.detach().numpy().squeeze(), ax=ax[1], title='prediction')
        plt.tight_layout()
        plt.show()

