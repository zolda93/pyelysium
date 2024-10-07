from elysium import empty
from .functional import embedding
from .parameter import Parameter
from . import init

class Embedding:
    def __init__(self,num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.,
                 scale_grad_by_freq=False, sparse=False, _weight=None,device='cpu'):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx>0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx<0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(empty((num_embeddings, embedding_dim))).to(device)
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings,
                                           embedding_dim], 'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)
        self.sparse = sparse
    def reset_parameters(self):
        init.normal_(self.weight)
        if self.padding_idx is not None:self.weight.data[self.padding_idx] = 0
    def __call__(self,x):return embedding(x,self.weight,padding_idx=self.padding_idx, max_norm=self.max_norm, norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse)
        

