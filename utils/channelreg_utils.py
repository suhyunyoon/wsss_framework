import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

# Custom Pooling Layer.
# Calculate by Sliding windows
class CustomPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False, func=lambda x:x.mean(dim=-1)):
        super(CustomPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same
        self.func = func

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
        return x

# 0-1 MinMax scaling
def minmax_scaling(x):
    size_len = len(x.size())
    xmin = x.amin(dim=tuple(i for i in range(2, size_len)), keepdim=True)
    xmax = x.amax(dim=tuple(i for i in range(2, size_len)), keepdim=True)

    return ((x - xmin) / (xmax - xmin))

def get_variance(x, norm=True):
    if norm:
        x = minmax_scaling(x)
    return x.var(dim=-1)

def get_product(x, norm=True):
    if norm:
        x = minmax_scaling(x)
    return x.prod(dim=-1) + 0.000001

def get_l1(x, norm=False):
    if norm:
        x = minmax_scaling(x)
    return x.abs().sum()

def get_l2(x, norm=False):
    if norm:
        x = minmax_scaling(x)
    return x.pow(2.0).sum()