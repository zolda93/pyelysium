from .functional import dropout
class Dropout:
    def __init__(self,p=0.5,inplace=False,training=True):
        self.p = p
        self.training=training
        self.inplace=inplace
    def __call__(self,x):return dropout(x,p=self.p,inplace=self.inplace,training=self.training)

