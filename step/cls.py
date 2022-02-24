# Training Classification Model

import os
from tqdm import tqdm
from datetime import datetime
import time

import torch
from torch import nn

from data.classes import get_voc_class #, get_voc_colormap, get_imagenet_class

# from torchvision.datasets import VOCSegmentation, VOCDetection
from data.datasets import voc_train_dataset, voc_val_dataset, voc_test_dataset
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

from utils.models import get_model
from utils.optims import get_cls_optimzier
from utils.metrics import eval_multilabel_metric
from utils.misc import TensorBoardLogger

import logging
logger = logging.getLogger('main')

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
        acc, precision, recall, f1, ap, map = eval_multilabel_metric(labels, logits, average='samples')

    model.train()
    return val_loss, acc, precision, recall, f1, ap, map
    

def run(args):
    logger.info('Training Classifier...')

    # Initialize Tensorboard logger
    if args.use_tensorboard:
        tb_logger = TensorBoardLogger(args.log_path)

    # Count GPUs
    n_gpus = torch.cuda.device_count()
    logger.info(f'{n_gpus} GPUs Available.')

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

        # dataset
        dataset_train = voc_train_dataset(args, args.train_list, 'cls')
        dataset_val = voc_val_dataset(args, args.eval_list, 'cls')
        
        # Unlabeled dataset
        if args.labeled_ratio < 1.0 or args.train_ulb_list:
            dataset_train_ulb = voc_train_dataset(args, args.train_ulb_list, 'cls')
        else:
            dataset_train_ulb = None

    # # COCO2014
    # elif args.dataset == 'coco':
    #     pass
    # # Cityscapes
    # elif args.dataset == 'cityscapes':
    #     pass

    logger.info(f'Train Dataset Length: {len(dataset_train)}')
    logger.info(f'Validation Dataset Length: {len(dataset_val)}')
    if dataset_train_ulb is not None:
        logger.info(f'Unlabeled Train Dataset Length: {len(dataset_train_ulb)}')
    
    # Dataloader
    #train_sampler = DistributedSampler(dataset_train)
    #val_sampler = DistributedSampler(dataset_val)
    train_dl = DataLoader(dataset_train, batch_size=args.train['batch_size'], num_workers=args.num_workers, 
                          shuffle=True, sampler=None, pin_memory=True)
    val_dl = DataLoader(dataset_val, batch_size=args.eval['batch_size'], num_workers=args.num_workers, 
                        shuffle=False, sampler=None, pin_memory=True)
    
    # Unlabeled dataloader
    if args.labeled_ratio < 1.0 or args.train_ulb_list:
        train_ulb_dl = DataLoader(dataset_train_ulb, batch_size=args.train['batch_size'],
                                  num_workers=args.num_workers, shuffle=True, sampler=None, pin_memory=True)


    # Get Model
    model = get_model(args.network, pretrained=True, num_classes=voc_class_num-1)
    
    # Optimizer
    optimizer, scheduler = get_cls_optimzier(args, model)
 
    # model dataparallel
    model = torch.nn.DataParallel(model).cuda()
    # model DDP(TBD)
    #model = torch.nn.parallel.DistributedDataParallel(model.cuda())

    # Loss (MultiLabelSoftMarginLoss or BCEWithLogitsLoss)
    class_loss = getattr(nn, args.train['loss'])(reduction='none').cuda()


    # Training 
    best_acc = 0.0
    for e in range(args.train['epochs']):
        model.train()
        
        tb_dict = {}
        train_loss = 0.
        logits, labels = [], []
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
        train_loss /= len(dataset_train)
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0) 
        
        # Logging
        acc, precision, recall, f1, _, map = eval_multilabel_metric(labels, logits, average='samples')
        logger.info('Epoch %d Train Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f' % (e+1, train_loss, map, acc, precision, recall))
        #logger.info(optimizer.state_dict)
        tb_dict['train/acc'] = acc
        tb_dict['train/precision'] = precision
        tb_dict['train/recall'] = recall
        tb_dict['train/f1'] = f1
        tb_dict['train/map'] = map
        tb_dict['lr'] = optimizer.param_groups[0]['lr'] # Need modification for other optims except SGDs

        # Validation (+ Final Validation)
        if e % args.verbose_interval == 0 or e+1 == args.train['epochs']:
            val_loss, val_acc, val_precision, val_recall, val_f1, val_ap, val_map = validate(model, val_dl, dataset_val, class_loss)
            logger.info('Validation Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f' % (val_loss, val_map, val_acc, val_precision, val_recall))
            tb_dict['eval/acc'] = val_acc
            tb_dict['eval/precision'] = val_precision
            tb_dict['eval/recall'] = val_recall
            tb_dict['eval/f1'] = val_f1
            tb_dict['eval/map'] = val_map
            # Save Best Model
            if val_acc >= best_acc:
                best_model_path = os.path.join(args.log_path, 'best.pth')
                torch.save(model.module.state_dict(), best_model_path)
                logger.info(f'{best_model_path} Saved.')
                best_acc = val_acc
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Update Tensorboard log
        tb_logger.update(tb_dict, e)

    # Save final model (split module from dataparallel)
    final_model_path = os.path.join(args.log_path, 'final.pth')
    torch.save(model.module.state_dict(), final_model_path)
    logger.info(f'{final_model_path} Saved.')
    
    logger.info('Done Finetuning.\n')

    return None