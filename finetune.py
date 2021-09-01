import torch
from torch import nn, optim
from tqdm import tqdm

from data.classes import get_voc_class, get_voc_colormap, get_imagenet_class

from torchvision.datasets import VOCSegmentation, VOCDetection
from data.datasets import VOCClassification
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor 

from model import get_model

if __name__=='__main__':
    #sadfasdfasdfasdf

    # Hyperparameters
    seed = 42
    batch_size = 32
    num_workers = 8
    epochs = 1

    # transformation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #h,w = 520, 520
    #h,w = 256, 256 -> RandomCrop 224
    h,w = 224, 224
    
    transform_train = Compose([Resize((h,w)),
                               ToTensor(),
                                #RandomHorizontalFlip(p=0.5),
                                Normalize(mean, std)])
    
    transform_val = Compose([Resize((h,w)),
                            ToTensor(),
                            Normalize(mean, std)])

    transform_target = Compose([Resize((h,w))])
                               #ToTensor()])

    # Get dataset & dataloader
    dataset_type = 'voc'
    
    if dataset_type == 'voc':
        voc_class = get_voc_class()
        voc_class_num = len(voc_class)
        #dataset = VOCSegmentation(root='/home/suhyun/dataset/VOC/', year='2012', image_set='train', download=False, transform=transform_train, target_transform=transform_target)
        dataset = VOCClassification(root='/home/suhyun/dataset/VOC/', year='2012', image_set='train', download=False, transform=transform_train)

    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)


    # Get Model
    model_name = 'resnet50'
    model_type = 'resnet'

    model = get_model(model_name, pretrained=True)

    # Switch FC layer
    if model_type == 'resnet':
        #in_features = model.fc.in_features
        in_features = 2048
        model.fc = nn.Linear(in_features, voc_class_num-1)

    # Optimizer
    class_loss = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()
    '''optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)'''
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

    # Training(Finetuning)
    losses = []
    accs = []

    model.cuda()
    for e in range(1, epochs+1):
        model.train()
        train_loss = 0.
        corrects = 0
        for img, label in tqdm(dl):
            img, label = img.cuda(), label.cuda()
    
            logits = model(img)
    
            loss = class_loss(logits, label).mean()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # loss ,acc
            train_loss += loss.detach().cpu()
            corrects += (label == logits.detach()).sum()
        train_loss /= len(dataset)
        acc = corrects / len(dataset)
        print('epoch %d Train Loss: %.6f, Accuracy: %.6f' % (e, train_loss, acc))
        # Update loss, acc
        losses.append(train_loss)
        accs.append(acc)
    print('Best Loss: %.6f' % min(losses))
    print('Best Acc: %.6f' % max(accs))

    # Evaluate CAM
    from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import numpy as np

    model.eval()
    # ResNet
    if model_type == 'resnet':
        target_layer = model.layer4[-1]
    # EfficientNet
    # ViT
    # Swin
    # ...
        
    # Function which makes CAM
    make_cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=True)

    # dataset
    dataset = VOCSegmentation(root='/home/suhyun/dataset/VOC/', year='2012', image_set='train', download=False, transform=transform_train, target_transform=transform_target)

    # Make CAM
    cam_eval_th = 0.15
    
    segs=[]
    preds=[]
    
    for i in tqdm(range(len(dataset))):
        pack = dataset[i]
        img = pack[0].cuda()
        seg = np.array(pack[1], dtype=np.uint8)
    
        # get image classes
        label = np.unique(seg)
        label = np.intersect1d(np.arange(voc_class_num-1), label)
    
        img = img.unsqueeze(0).repeat(len(label),1,1,1)
    
        pred_cam = make_cam(input_tensor=img, target_category=label)
    
        # Add background
        label = np.pad(label+1, (1,0), mode='constant', constant_values=0)
        pred_cam = np.pad(pred_cam, ((1,0),(0,0),(0,0)), mode='constant', constant_values=cam_eval_th)
        pred = np.argmax(pred_cam, axis=0)
        pred = label[pred]
    
        # Append
        segs.append(seg)
        preds.append(pred)

    # Evaluate mIoU
    from chainercv.evaluations import calc_semantic_segmentation_confusion
    confusion = calc_semantic_segmentation_confusion(preds, segs)
    
    print(confusion.shape)
    
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    
    for k, i in zip(voc_class, iou):
        print('%-15s:' % k,  i)
    print('%-15s:' % 'miou', np.nanmean(iou))
