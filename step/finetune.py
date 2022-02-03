import os
from tqdm import tqdm
from datetime import datetime
import time

import torch
from torch import nn

from data.classes import get_voc_class #, get_voc_colormap, get_imagenet_class

# from torchvision.datasets import VOCSegmentation, VOCDetection
from data.datasets import VOCClassification, get_transform
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

from utils.models import get_model
from utils.optims import get_finetune_optimzier
from utils.metrics import eval_multilabel_metric

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
        acc, precision, recall, f1, _, map = eval_multilabel_metric(labels, logits, average='samples')
        print('Validation Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1: %.2f' % (val_loss, map, acc, precision, recall, f1))

    model.train()
    return val_loss, map, acc, precision, recall, f1
    

def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Finetuning...')

    # Count GPUs
    n_gpus = torch.cuda.device_count()

    # Seed (reproducibility)
    # import random
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # cudnn.deterministic = True

    # Dataset
    # VOC2012
    if args.dataset == 'voc12':
        voc_class = get_voc_class()
        voc_class_num = len(voc_class)
        # transform
        transform_train = get_transform('train', args.train['crop_size'])
        transform_val = get_transform('val', args.eval['crop_size'])
        # dataset
        dataset_train = VOCClassification(root=args.dataset_root, year='2012', image_set='train', 
                                            dataset_list=args.train_list, download=False, transform=transform_train)
        dataset_val = VOCClassification(root=args.dataset_root, year='2012', image_set='val', 
                                            dataset_list=args.eval_list, download=False, transform=transform_val)
        
        # Unlabeled dataset
        if args.labeled_ratio < 1.0:
            pass

    # COCO2014
    elif args.dataset == 'coco':
        pass

    # Cityscapes
    elif args.dataset == 'cityscapes':
        pass
    
    # Dataloader
    #train_sampler = DistributedSampler(dataset_train)
    #val_sampler = DistributedSampler(dataset_val)
    train_dl = DataLoader(dataset_train, batch_size=args.train['batch_size'], num_workers=args.num_workers, 
                            shuffle=True, sampler=None, pin_memory=True)
    val_dl = DataLoader(dataset_val, batch_size=args.eval['batch_size'], num_workers=args.num_workers, 
                            shuffle=False, sampler=None, pin_memory=True)
    
    # Unlabeled dataloader
    if args.labeled_ratio < 1.0:
        pass

    # Get Model
    model = get_model(args.network, pretrained=True, num_classes=voc_class_num-1)
    
    # Optimizer
    optimizer, scheduler = get_finetune_optimzier(args, model)
 
    # model dataparallel
    model = torch.nn.DataParallel(model).cuda()
    # model DDP(TBD)
    #model = torch.nn.parallel.DistributedDataParallel(model.cuda())

    # Loss
    class_loss = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()
    #class_loss = nn.BCEWithLogitsLoss(reduction='none').cuda()

    # Logging dir
    args.log_path = os.path.join(args.log_dir, args.log_name)
    if os.path.exists(args.log_path):
        # Overwrite existing dir
        if args.log_overwrite:
            import shutil
            shutil.rmtree(args.log_path)
            os.mkdir(args.log_path)

        # Make another directory
        else:
            cur_time = str(int(time.time()))
            args.log_path += '_' + cur_time
            args.log_name += '_' +str(int(time.time()))
    
    else:
        # Make log directory
        os.mkdir(args.log_path)  
        
    print('Log Path:', args.log_path)

    # Training 
    best_acc = 0.0
    for e in range(1, args.train['epochs']+1):
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
            nth_logit = torch.sigmoid(logit).detach()
            logits.append(nth_logit)
            
        # Training log
        # loss
        train_loss /= len(dataset_train)
        # eval
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0) 
        acc, precision, recall, f1, _, map = eval_multilabel_metric(labels, logits, average='samples')
        print('epoch %d Train Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1: %.2f' % (e, train_loss, map, acc, precision, recall, f1))
        
        # Validation
        if e % args.verbose_interval == 0:
            _, _, val_acc, _, _, _ = validate(model, val_dl, dataset_val, class_loss)
            # Save Best Model
            if val_acc >= best_acc:
                torch.save(model.module.state_dict(), os.path.join(args.log_path, 'best.pth'))
                best_acc = val_acc

    # Final Validation
    if e % args.verbose_interval != 0:
        validate(model, val_dl, dataset_val, class_loss)

    # Save final model (split module from dataparallel)
    final_model_path = os.path.join(args.log_path, 'final.pth')
    torch.save(model.module.state_dict(), final_model_path)
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done Finetuning.')
    print()

    return None
