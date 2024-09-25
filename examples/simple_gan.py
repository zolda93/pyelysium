# for now run the code in google colab
#!git clone https://github.com/zolda93/pyelysium.git
#%cd /content/pyelysium
#!pip install torch_snippets

import elysium
from elysium import nn
from elysium import optim
# i use torch dataset and dataloader since elysium doesn't have them yet!
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize
from torchvision.utils import make_grid
from torch_snippets.torch_loader import Report
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_transform = Compose([
    ToTensor(),
    Normalize(mean=(0.5),std=(0.5)),
])
training_data = MNIST('./mnist/',transform=img_transform,train=True,download=True)
class Discriminator:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(28*28,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid())
    def __call__(self,x):
        return self.model(x)
class Generator:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(100,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,28*28),
            nn.Tanh(),)
    def __call__(self,x):
        return self.model(x)
class GAN:
    def __init__(self,discr,gener):
        self.discriminator = discr
        self.generator = gener
        self.data_loader = DataLoader(training_data,batch_size=128,shuffle=True)
        self.Doptimizer = optim.Adam(self.discriminator, lr=0.0002)
        self.Goptimizer = optim.Adam(self.generator, lr=0.0002)
        self.loss = nn.BCELoss()
    def train_discriminator(self,real_data,fake_data):
        # reset gradient
        self.Doptimizer.zero_grad()
        #train discriminator with real data labele as one
        pred_real = self.discriminator(real_data)
        error_real = self.loss(pred_real,elysium.ones((len(real_data),1)))
        error_real.backward()
        #train discriminator with fake data labeled as zero
        pred_fake = self.discriminator(fake_data)
        error_fake = self.loss(pred_fake,elysium.zeros((len(fake_data),1)))
        error_fake.backward()
        #update
        self.Doptimizer.step()
        return error_real + error_fake
    def train_generator(self,fake_data,real_data):
        self.Goptimizer.zero_grad()
        pred = self.discriminator(fake_data)
        error = self.loss(pred,elysium.ones((len(real_data),1)))
        error.backward()
        self.Goptimizer.step()
        return error
    def train_gan(self):
        num_epochs = 20
        log = Report(num_epochs)
        for e in range(num_epochs):
            N = len(self.data_loader)
            for idx,(images,_) in enumerate(self.data_loader):
                real = elysium.tensor(images.view(len(images),-1).numpy())
                fake = self.generator(elysium.randn((len(real),100)))
                fake = fake.detach()
                d_loss = self.train_discriminator(real,fake)
                fake = self.generator(elysium.randn((len(real),100)))
                g_loss = self.train_generator(fake,real)
                log.record(e+(1+idx)/N, d_loss=d_loss.item(), g_loss=g_loss.item(), end='\r')
            log.report_avgs(e+1)
        log.plot_epochs(['d_loss', 'g_loss'])


if __name__ == '__main__':
    gan = GAN(Discriminator(),Generator())
    gan.train_gan()
    z = elysium.tensor(torch.randn(64, 100).numpy())
    sample_images = gan.generator(z)
    sample_images = torch.tensor(sample_images.detach().numpy().reshape(64, 1, 28, 28))
    grid = make_grid(sample_images, nrow=8, normalize=True)
    from torch_snippets import *
    show(grid.cpu().detach().permute(1,2,0), sz=5)
