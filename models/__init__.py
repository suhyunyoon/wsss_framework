from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.resnet import resnext50_32x4d, resnext101_32x8d
from models.resnet import wide_resnet50_2, wide_resnet101_2
from models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
#from models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

# Channel-wise regularization
from models.channelreg_resnet import resnet18 as creg_resnet18
from models.channelreg_resnet import resnet34 as creg_resnet34
from models.channelreg_resnet import resnet50 as creg_resnet50
from models.channelreg_resnet import resnet101 as creg_resnet101
from models.channelreg_resnet import resnet152 as creg_resnet152