import torch.nn

from torch.nn import MultiLabelSoftMarginLoss, BCEWithLogitsLoss, BCELoss, MultiLabelMarginLoss 

from torch.nn import CrossEntropyLoss, MSELoss, MultiMarginLoss
from torch.nn import NLLLoss, NLLLoss2d

from utils.loss.losses import AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel