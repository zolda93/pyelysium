import os
class Dataset:
    def __init__(self,root,train=True,transforms=None,target_transform=None,download=False):
        self.root = root + '/{}'.format(self.__class__.__name__.lower())
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
    def __len__(self):raise NotImplementedError
    def __getitem__(self,idx):raise NotImplementedError

        
