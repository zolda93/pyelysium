from . import functional as F
class L1Loss:
    def __init__(self,reduction='mean'):
        self.reduction = reduction
    def __call__(self,x,target):
        return F.l1_loss(x,target,reduction=self.reduction)
class MSELoss:
    def __init__(self,reduction='mean'):
        self.reduction = reduction
    def __call__(self,x,target):
        return F.mse_loss(x,target,reduction=self.reduction)
class BCELoss:
    def __init__(self,reduction='mean',weight=None):
        self.reduction = reduction
        self.weight = weight
    def __call__(self,x,target):
        return F.binary_cross_entropy(x,target,weight=self.weight,reduction=self.reduction)
class BCEWithLogitsLoss:
    def __init__(self,reduction='mean',weight=None,pos_weight=None):
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
    def __call__(self,x,target):
        return F.binary_cross_entropy_with_logits(x,target,weight=self.weight,pos_weight=self.pos_weight,reduction=self.reduction)
class NLLLoss:
    def __init__(self,weight=None,ignore_index=-100,reduction='mean'):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    def __call__(self,x,target):
        return F.nll_loss(x, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
class CrossEntropyLoss:
    def __init__(self,weight=None,ignore_index=-100,reduction='mean'):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    def __call__(self,x,target,axis=1):
        return F.cross_entropy(x,target,axis=axis,weight=self.weight,ignore_index=self.ignore_index,reduction=self.reduction)
