import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from torch.nn.modules.utils import _pair, _quadruple

# Custom Pooling Layer.
# Calculate by Sliding windows
class CustomPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False, func=lambda x:x.mean(dim=-1), get_loss=True):
        super(CustomPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same
        self.func = func
        self.get_loss = get_loss

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,))
        # Forward Function at dim=-1 (input:(N,C,H,W,Group) -> output:(N,C,H,W))
        # EX) self.func = lambda x: x.median(dim=-1)[0]
        x = self.func(x)

        if self.get_loss:
            feat_dim = tuple(i for i in range(1, len(x.size())))
            return x.sum(dim=feat_dim)
        else:
            return x


# Orthogonality(feature spatial | channel)
class Orthogonality(nn.Module):
    def __init__(self, target='spatial', sort_by='mean', k=128, symmetric=False):
        super(Orthogonality, self).__init__()
        self.target = target
        self.sort_by = sort_by
        self.k = k
        self.symmetric = symmetric

    # weight도 41
    def get_orth_reg(self, x):
        x = minmax_scaling(x, start_dim=1)
        # Orthogonality
        reg = x.transpose(-1,-2) @ x
        # Norm
        #return LA.norm(reg - torch.eye(reg.size(1), device=reg.device), dim=(1,2)) # Frobenius(L2) norm
        reg = reg - reg.detach().diagonal(dim1=-2, dim2=-1).diag_embed()
        return LA.norm(reg, dim=(1,2)) # Frobenius(L2) norm

    def forward(self, x):
        x = x.flatten(2)
        x = minmax_scaling(x, start_dim=1)

        # 상위 K개의 element(channel 혹은 pixel representation) 사용
        # Sort by mean
        if self.sort_by == 'mean':
            x_sorted = x.mean(dim=-1)
            x_idx = torch.argsort(x_sorted, dim=-1, descending=True)
        # Sort by max
        elif self.sort_by == 'max':
            x_sorted = x.max(dim=-1).values
            x_idx = torch.argsort(x_sorted, dim=-1, descending=True)
        # 상위 K개
        x_ = x[:,x_idx[:self.k]]

        # pixel 별 representation이 orthogonal 하도록(USELESS)
        if self.target == 'spatial':
            reg = self.get_orth_reg(x_)

        # Channel 별 feature map이 orthogonal 하도록(다른 영역을 보도록)
        elif self.target == 'channel':
            reg = self.get_orth_reg(x_.transpose(-1,-2))

        # if self.symmetric:
        #     reg += self.get_orth_reg(x.transpose(-1,-2))
            
        return reg


# 0-1 MinMax scaling
def minmax_scaling(x, start_dim=2):
    end_dim = len(x.size())
    xmin = x.amin(dim=tuple(i for i in range(start_dim, end_dim)), keepdim=True)
    xmax = x.amax(dim=tuple(i for i in range(start_dim, end_dim)), keepdim=True)

    return ((x - xmin) / (xmax - xmin))

# For Custom Pooling
def get_variance(x, norm=True):
    if norm:
        x = minmax_scaling(x)
    return x.var(dim=-1)

def get_product(x, norm=True):
    if norm:
        x = minmax_scaling(x)
    return x.prod(dim=-1) + 0.000001

def get_l1(x, norm=True):
    if norm:
        x = minmax_scaling(x)
    return x.abs().sum()

def get_l2(x, norm=True):
    if norm:
        x = minmax_scaling(x)
    return x.pow(2.0).sum()

# Feature orthogonality
def get_feature_orthogonality(feat):

    return feat