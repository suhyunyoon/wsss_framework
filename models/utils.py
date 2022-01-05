import torch
from torch import nn

import timm

import os
import importlib

#from models import resnet18, resnet34, resnet50, resnet101, resnet152
#from models import resnext50_32x4d, resnext101_32x8d
#from models import wide_resnet50_2, wide_resnet101_2
#from models import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, vgg16_bn_sl, vgg19_bn_sl
# Deit
# ...


# Get model and architecture type(str)
def get_model(model_name, pretrained=False, num_classes=1000):
    # External Model: IRN (20 classes only)
    if model_name == 'irn.net.resnet50_cam':
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
    # ViT(Vision Transformer)
    elif 'vit' in model_name:        
        # Self-supervsied pretrained model DINO(vits,vitb)
        if model_name.startswith('dino_'):
            if model_name == 'dino_vits16':
                model = timm.create_model('vit_small_patch16_224')(num_classes=num_classes)
            elif model_name == 'dino_vits8':
                model = timm.create_model('vit_small_patch8_224')(num_classes=num_classes)
            elif model_name == 'dino_vitb16':
                model = timm.create_model('vit_base_patch16_224')(num_classes=num_classes)
            elif model_name == 'dino_vitb8':
                model = timm.create_model('vit_base_patch8_224')(num_classes=num_classes)

            pretrained = torch.hub.load('facebookresearch/dino:main', model_name)
            model.load_state_dict(pretrained.state_dict(), strict=False)
            # xcit 
            #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8', pretrained=pretrained)
        
        # Official timm vit
        else:
            model = timm.create_model(model_name)(pretrained=True, num_classes=num_classes)
    
    
    # Scratch Supervised pretrained CNN models
    else:
        model = getattr(importlib.import_module('models'), model_name)(pretrained=pretrained, num_classes=num_classes)

    return model

# Return target layer which extracts CAM
def get_cam_target_layer(args, model):
    # Resnet
    if hasattr(model, 'layer4'): 
        return model.layer4[-1]
    # irn resnet
    elif hasattr(model, 'stage4'): 
        return model.stage4[-1]
    # VGG
    elif hasattr(model, 'extra'):
        return model.extra[-2]
    # ViTs
    elif hasattr(model, 'blocks'):
        target_layer = model.blocks[-1].norm1
    # Swin-Transformer
    elif hasattr(model, 'layers') and hasattr(model.layers[-1], 'block'):
        return model.layers[-1].block[-1].norm1
    # EfficientNet
    elif hasattr(model, 'features'):
        return model.features[-1]
    else:
        return None
