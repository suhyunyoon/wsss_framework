import torch
from torch import nn, optim
import os
from tqdm import tqdm
from datetime import datetime

from data.classes import get_voc_class, get_voc_colormap, get_imagenet_class

from torchvision.datasets import VOCSegmentation, VOCDetection
from data.datasets import VOCClassification, get_transform
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor 

from models.utils import get_model, Classifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

# for evaluation
#from sklearn.metrics import classification_report, multilabel_confusion_matrix
def eval_multilabel_metric(label, logit, average="samples"):
    pred = logit >= 0.5

    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, average=average, zero_division=0)
    recall = recall_score(label, pred, average=average)
    f1 = f1_score(label, pred, average=average)
    
    return acc, precision, recall, f1

# Validation in training
def validate(model, dl, dataset, class_loss):
    model.eval()
    with torch.no_grad():
        val_loss = 0.
        logits = []
        labels = []
        for img, label in tqdm(dl):
            # memorize labels
            labels.append(label)
            img, label = img.cuda(), label.cuda()
            
            # calc loss
            logit = model(img)
            loss = class_loss(logit, label).mean()
            
            # loss
            val_loss += loss.detach().cpu()
            # acc
            logit = torch.sigmoid(logit).detach()
            logits.append(logit)
        # Eval
        # loss
        val_loss /= len(dataset)
        # eval
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0)
        acc, precision, recall, f1 = eval_multilabel_metric(labels, logits, average='samples')
        print('Validation Loss: %.6f, Accuracy: %.6f, Precision: %.6f, Recall: %.6f, F1: %.6f' % (val_loss, acc, precision, recall, f1))

    model.train()
    

def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Finetuning...')

    # Count GPUs
    n_gpus = torch.cuda.device_count()

    # Dataset
    # VOC2012
    if args.dataset == 'voc12':
        voc_class = get_voc_class()
        voc_class_num = len(voc_class)
        # transform
        transform_train = get_transform('train', args.crop_size)
        transform_val = get_transform('val', args.crop_size)
        # dataset
        dataset_train = VOCClassification(root=args.voc12_root, year='2012', image_set=args.train_set, download=False, transform=transform_train)
        dataset_val = VOCClassification(root=args.voc12_root, year='2012', image_set=args.eval_set, download=False, transform=transform_val)

    # else
    else:
        pass
    
    # Dataloader
    #train_sampler = DistributedSampler(dataset_train)
    #val_sampler = DistributedSampler(dataset_val)
    train_dl = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=None)
    val_dl = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=None)

    # Get Model + Switch FC layer
    model, model_type = get_model(args.network, pretrained=True, classifier=True, class_num=voc_class_num-1)

    weights_path = os.path.join(args.weights_dir, args.network + '.pth')
    #model.load_state_dict(torch.load(weights_path), strict=True)

    # Loss
    class_loss = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()
    #class_loss = nn.BCEWithLogitsLoss(reduction='none').cuda()
    
    # Optimizer
    if model_type == 'vits' or model_type == 'vitb':
        optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.04)
    #elif args.network == 'dino_resnet50':
    #    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # elif model_type == 'resnet':
    #     param_groups = model.module.get_parameter_groups()
    #     optimizer = optim.SGD([
    #         {'params': param_groups[0], 'lr': args.lr},
    #         {'params': param_groups[1], 'lr': 2*args.lr},
    #         {'params': param_groups[2], 'lr': 10*args.lr},
    #         {'params': param_groups[3], 'lr': 20*args.lr}], 
    #         momentum=0.9, 
    #         weight_decay=0.04, 
    #         nesterov=True
    #     )
    # PolyOptimizer?
    elif model_type == 'vgg':
        param_groups = model.get_parameter_groups()
        optimizer = optim.SGD([
            {'params': param_groups[0], 'lr': args.learning_rate},
            {'params': param_groups[1], 'lr': 2*args.learning_rate},
            {'params': param_groups[2], 'lr': 10*args.learning_rate},
            {'params': param_groups[3], 'lr': 20*args.learning_rate}], 
            momentum=0.9, 
            weight_decay=0.0005, 
            nesterov=True
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    # Scheduler
    ############
 
    # model dataparallel
    model = torch.nn.DataParallel(model).cuda()
    # model DDP
    # ...
    #model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    model.train()

    # Training 
    for e in range(1, args.epochs+1):
        model.train()
        
        train_loss = 0.
        logits = []
        labels = []
        for img, label in tqdm(train_dl):
            
            # memorize labels
            labels.append(label)
            img, label = img.cuda(), label.cuda()
            
            # calc loss
            logit = model(img)
        
            loss = class_loss(logit, label).mean()
            # training
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss
            train_loss += loss.detach().cpu()
            # acc
            logit = torch.sigmoid(logit).detach()
            logits.append(logit)
            
        # Training log
        if e % args.verbose_interval != 0:
            # loss
            train_loss /= len(dataset_train)
            # eval
            logits = torch.cat(logits, dim=0).cpu()
            labels = torch.cat(labels, dim=0) 
            acc, precision, recall, f1 = eval_multilabel_metric(labels, logits, average='samples')
            print('epoch %d Train Loss: %.6f, Accuracy: %.6f, Precision: %.6f, Recall: %.6f, F1: %.6f' % (e, train_loss, acc, precision, recall, f1))
        
        # Validation
        if e % args.verbose_interval == 0:
            validate(model, val_dl, dataset_val, class_loss)
    
    # Save final model
    if args.weights_name is None:
        weights_path = os.path.join(args.weights_dir, f'{args.network}_e{args.epochs}_b{args.batch_size}.pth')
    else:
        weights_path = os.path.join(args.weights_name, args.weights_name)
    # split module from dataparallel
    torch.save(model.module.state_dict(), weights_path)
    torch.cuda.empty_cache()
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done Finetuning.')
    print()

