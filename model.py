import torch
from torch import nn

from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnext101_32x8d

resnets = 'resnet'
vits    = 'vits'
vitb    = 'vitb'
# Eff
# Swin
# Deit
# ...


# Get model and architecture type(str)
def get_model(model_name='resnet50', pretrained=False, classifier=False, class_num=1000):
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
    elif model_name == 'dino_vits16':
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

    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p8', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16', pretrained=pretrained)
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8', pretrained=pretrained)
    #model_type = 'xcit'

    elif model_name == 'dino_resnet50':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=pretrained)
        model_type = resnets
    # External Model: IRN
    elif model_name == 'external.irn.net.resnet50_cam':
        model = getattr(importlib.import_module(args.cam_network), 'CAM')()
        if pretrained:
            model.load_state_dict(torch.load(model_name + '.pth'), strict=True)
        model_type = resnets
    else:
        print('Model', model_name, 'Not Exists!')
        model = None
        model_type = ''

    # return Classifier
    if classifier:
        model = Classifier(model, model_type, class_num)

    return model, model_type

# Return target layer which extracts CAM
def get_cam_target_layer(model, model_type):
    # get target layer
    # from model.model where model is ./Classifier
    if model_type == 'resnet': 
        target_layer = model.layer4[-1] if hasattr(model,'layer4') else model.model.layer4[-1]
    elif model_type == 'vits' or model_type == 'vitb':
        target_layer = model.blocks[-1].norm1 if hasattr(model, 'blocks') else model.model.blocks[-1].norm1
    elif model_type == 'swin':
        target_layer = model.layers[-1].block[-1].norm1 if hasattr(model, 'blocks') else model.model.blocks[-1].norm1
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
        if model_type == 'resnet':
            f_num = model.fc.in_features if hasattr(model.fc, 'in_features') else 2048
            model.fc = nn.Identity()
        elif model_type == 'vits' or model_type == 'vitb':
            f_num = {'vits': 384, 'vitb': 768}
            f_num = f_num[model_type]
            model.head = nn.Identity()
        elif model_type == 'swin':
            f_num = -1
            model.head = nn.Identity()
        else:
            f_num = -1
        
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

