import torch
import timm

import os
import importlib

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

    # Self-supervsied pretrained model DINO(vits,vitb)
    elif model_name.startswith('dino_'):
        if model_name == 'dino_vits16':
            model = timm.create_model('vit_small_patch16_224')(num_classes=num_classes)
        elif model_name == 'dino_vits8':
            model = timm.create_model('vit_small_patch8_224')(num_classes=num_classes)
        elif model_name == 'dino_vitb16':
            model = timm.create_model('vit_base_patch16_224')(num_classes=num_classes)
        elif model_name == 'dino_vitb8':
            model = timm.create_model('vit_base_patch8_224')(num_classes=num_classes)
        elif model_name == 'dino_resnet50':
            model = getattr(importlib.import_module('models'), 'resnet50')(pretrained=False, num_classes=num_classes)

        pretrained = torch.hub.load('facebookresearch/dino:main', model_name)
        model.load_state_dict(pretrained.state_dict(), strict=False)
        # xcit 
        #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8', pretrained=pretrained)
    
    # ViT(Vision Transformer)
    elif 'vit' in model_name:
        model = timm.create_model(model_name)(pretrained=pretrained, num_classes=num_classes)
    
    # Scratch Supervised pretrained CNN models
    else:
        model = getattr(importlib.import_module('models'), model_name)(pretrained=pretrained, num_classes=num_classes)

    return model

# Return target layer which extracts CAM
def get_cam_target_layer(model):

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


# reshape_transform for vits and swin
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), 
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_7(tensor, height=7, width=7):
    return reshape_transform(tensor, height, width)

def reshape_transform_14(tensor, height=14, width=14):
    return reshape_transform(tensor, height, width)

def reshape_transform_28(tensor, height=28, width=28):
    return reshape_transform(tensor, height, width)

def get_reshape_transform(model_name):
    if model_name in []:
        target_tr = reshape_transform_7
    elif model_name in ['dino_vits16', 'dino_vitb16', 'swin']:
        target_tr = reshape_transform_14
    elif model_name in ['dino_vits8', 'dino_vitb8']:
        target_tr = reshape_transform_28
    else:
        target_tr = None
    return target_tr