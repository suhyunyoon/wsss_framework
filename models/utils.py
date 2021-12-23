import torch
from torch import nn

import os
import importlib

from models import resnet18, resnet34, resnet50, resnet101, resnet152
from models import resnext50_32x4d, resnext101_32x8d
from models import wide_resnet50_2, wide_resnet101_2
from models import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

resnets = 'resnet'
vggs = 'vgg'
efficientnets = 'efficientnet'
vits    = 'vits'
vitb    = 'vitb'
swin    = 'swin'
# Deit
# ...


# Get model and architecture type(str)
def get_model(model_name='resnet50', pretrained=False, classifier=False, class_num=1000):
    # Self-supervsied pretrained model
    # DINO
    if model_name == 'dino_vits16':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained)
        model_type = vits
    elif model_name == 'dino_vits8':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=pretrained)
        model_type = vits
    elif model_name == 'dino_vitb16':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=pretrained)
        model_type = vitb
    elif model_name == 'dino_vitb8':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=pretrained)
        model_type = vitb
    # SSL CNN
    elif model_name == 'dino_resnet50':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=pretrained)
        model_type = resnets
    # External Model: IRN
    elif model_name == 'irn.net.resnet50_cam':
        model_name_split = model_name.split('.')
        # move current directory into external project
        importlib.sys.path[0] = os.path.join('external', model_name_split[0])
        # temporary model directory
        model_name_ = '.'.join(model_name_split[1:])
        model = getattr(importlib.import_module(model_name_), 'CAM')()
        importlib.sys.path[0] = ''
        # get pretrained path
        if pretrained:
            model.load_state_dict(torch.load(model_name + '.pth'), strict=True)
        model_type = model_name_split[0] + '.' + resnets
    # Scratch CNN models
    # Supervised pretrained model
    else:
        if model_name.startswith('resn') or model_name.startswith('wide_resnet'):
            model_type = resnets
        elif model_name.startswith('vgg'):
            model_type = vggs
        elif model_name.startswith('efficientnet'):
            model_type = efficientnets
        else:
            print('Model', model_name, 'Not Exists!')
            model = None
        # Get model class by 
        model = eval(model_name)(pretrained=pretrained)

    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p8', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8', pretrained=pretrained)
    #model_type = 'xcit'
    
    # return Classifier
    if classifier:
        model = Classifier(model, model_type, class_num)

    return model, model_type

# Return target layer which extracts CAM
def get_cam_target_layer(model, model_type):
    # get target layer
    # from model.model where model is ./Classifier
    if model_type == resnets: 
        target_layer = model.layer4[-1] if hasattr(model,'layer4') else model.model.layer4[-1]
    elif model_type == vggs:
        target_layer = model.features[-1] if hasattr(model,'features') else model.model.features[-1]
    elif model_type == efficientnets:
        pass
    elif model_type == vits or model_type == vitb:
        target_layer = model.blocks[-1].norm1 if hasattr(model, 'blocks') else model.model.blocks[-1].norm1
    elif model_type == swin:
        target_layer = model.layers[-1].block[-1].norm1 if hasattr(model, 'blocks') else model.model.blocks[-1].norm1
    elif model_type == 'irn.'+resnets:
        target_layer = model.stage4[-1][-1] if hasattr(model, 'stage4') else model.model.stage4[-1][-1]
    else:
        target_layer = None
    return target_layer

# Classifier with linear eval
class Classifier(nn.Module):
    def __init__(self, model, model_type, num_classes=20):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        # get backbone model
        self.model = model

        # remove existing linear
        if model_type == resnets:
            f_num = model.fc.in_features if hasattr(model.fc, 'in_features') else 2048
            model.fc = nn.Identity()
        ############################
        ### Wideresnet, Resnexts###
        #############################
        elif model_type == vggs:
            f_num = model.classifier[-1].in_features if hasattr(model, 'classifier') else 1000
            model.classifier[-1] = nn.Identity()

        elif model_type == efficientnets:
            f_num = -1

        elif model_type == vits or model_type == vitb:
            f_num = {vits: 384, vitb: 768}
            f_num = f_num[model_type]
            model.head = nn.Identity()

        elif model_type == swin:
            f_num = -1
            model.head = nn.Identity()

        # External models
        elif model_type == 'irn.'+resnets:
            f_num = -1
        else:
            f_num = -1
        
        # FC에 Linear 말고 다른 Layer를 사용 할 때
        if model_type == 'irn.'+resnets:
            self.fc = nn.Identity()
        else:
            # Linear
            self.fc = nn.Linear(f_num, num_classes)
            # weight init
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

