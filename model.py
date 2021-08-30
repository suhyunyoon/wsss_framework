import torch

resnets = 'resnet'
vits    = 'vit'
# Eff
# Swin
# Deit
# ...


# Get model and architecture type(str)
def get_model(model_name='resnet50', pretrained=False):
    # Supervised pretrained model
    if model_name == 'resnet34':
        model = resnet34(pretrained=pretrained)
        model_type = resnets
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        model_type = resnets
    elif model_name == 'resnet101':
        model = resnet101(pretrained=pretrained)
        model_type = resnets
    elif model_name == 'resnet152':
        model = resnet152(pretrained=pretrained)
        model_type = resnets
    elif model_name == 'resnet101_32x8d':
        model = resnet101_32x8d(pretrained=pretrained)
        model_type = resnets

    # Self-supervsied pretrained model
    # DINO
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=pretrained)
    #model_type = 'vit'
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p8', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8', pretrained=pretrained)
    #model_type = 'xcit'

    elif model_name == 'dino_resnet50':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=pretrained)
        model_type = resnets
    else:
        print('Model', model_name, 'Not Exists!')
        model = None
        model_type = ''

    return model, model_type

# Return target layer which extracts CAM
def get_cam_target_layer(model, model_type):
    if model_type == 'resnet':
        target_layer = model.layer4[-1]
    elif model_type == 'vit':
        target_layer = model.blocks[-1].norm1
    elif model_type == 'swin':
        target_layer = model.layers[-1].block[-1].norm1
    else:
        target_layer = None
    return target_layer
