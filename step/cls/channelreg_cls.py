# Training Classification Model

import os
from tqdm import tqdm

import torch
from torch import nn, multiprocessing
import torch.nn.functional as F

# from torchvision.datasets import VOCSegmentation, VOCDetection
from utils.datasets import voc_train_dataset, voc_val_dataset, voc_test_dataset
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

import utils.loss
from utils.models import get_model
from utils.optims import get_cls_optimzier

from utils.misc import TensorBoardLogger, make_logger
from utils.train import validate, eval_multilabel_metric

from utils.channelreg_utils import CustomPool2d, get_variance, get_product

# Channel-wise Regularization
from torchvision.models.feature_extraction import create_feature_extractor

import logging
logger = logging.getLogger('main')

# Seed (reproducibility)
    # import random
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # cudnn.deterministic = True

def _work(pid, args, dataset_train, dataset_val, dataset_train_ulb):
    logger, _ = make_logger(args, is_new=False)

    # Initialize Tensorboard logger
    if args.use_tensorboard:
        tb_logger = TensorBoardLogger(args.log_path)

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
    model = get_model(args.network, pretrained=True, num_classes=args.voc_class_num-1)

    # Optimizer
    optimizer, scheduler = get_cls_optimzier(args, model)
 
    # model dataparallel
    model = torch.nn.DataParallel(model).cuda()
    # model DDP(TBD)
    #model = torch.nn.parallel.DistributedDataParallel(model.cuda())

    # Loss (MultiLabelSoftMarginLoss or BCEWithLogitsLoss or etc..)
    class_loss = getattr(utils.loss, args.train['loss']['name'])(**args.train['loss']['kwargs']).cuda()

    # Training 
    best_acc = 0.0
    for e in range(args.train['epochs']):
        tb_dict = {}
        # Validation
        if e % args.verbose_interval == 0:
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

        model.train()

        # Channel-wise regularization
        if args.train['channelreg'] == 'variance':
            ch_func = get_variance
        elif args.train['channelreg'] == 'product':
            ch_func = get_product
        ch_pool = CustomPool2d(kernel_size=5, stride=1, padding=0, func=ch_func)

        train_loss = 0.
        cls_loss, channel_loss = 0., 0.
        logits, labels = [], []
        for img, label in tqdm(train_dl):
            # memorize labels
            labels.append(label)
            img, label = img.cuda(), label.cuda()
            
            # calc loss
            logit, features, cam = model(img)            
            loss = class_loss(logit, label).mean()
            cls_loss += loss.detach().cpu()

            # Channel-wise loss
            if e >= args.train['warmup_epochs']:
            #     Option 1 - Channel Maximization
            #     feature = features[-1]
            #     feature_pl = torch.max(feature.detach(), dim=1).values
            #     feature_pl = feature_pl.unsqueeze(dim=1).repeat(1,feature.size(1),1,1)
            #     channel_loss = F.mse_loss(feature, feature_pl)

            #     loss += args.train['lambda'] * channel_loss
                # Option 2
                feature = features[-1]
                feature = ch_pool(feature)
                feature_dim = tuple(i for i in range(1, len(feature.size())))
                chloss = feature.sum(dim=feature_dim)

                loss += args.train['lambda'] * chloss.mean()
                channel_loss += chloss.mean().detach().cpu()

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
        cls_loss /= len(dataset_train)
        channel_loss /= len(dataset_train)

        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0) 
        
        # Logging
        acc, precision, recall, f1, _, map = eval_multilabel_metric(labels, logits, average='samples')
        logger.info('Epoch %d Train Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f' % (e+1, train_loss, map, acc, precision, recall))
        #logger.info(optimizer.state_dict)
        tb_dict['train/loss'] = train_loss
        tb_dict['train/classification_loss'] = cls_loss
        tb_dict['train/channel_loss'] = channel_loss
        tb_dict['train/acc'] = acc
        tb_dict['train/precision'] = precision
        tb_dict['train/recall'] = recall
        tb_dict['train/f1'] = f1
        tb_dict['train/map'] = map
        tb_dict['lr'] = optimizer.param_groups[0]['lr'] # Need modification for other optims except SGDs
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Update Tensorboard log
        tb_logger.update(tb_dict, e)

    # Final Validation
    val_loss, val_acc, val_precision, val_recall, val_f1, val_ap, val_map = validate(model, val_dl, dataset_val, class_loss)
    logger.info('Final Validation Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f' % (val_loss, val_map, val_acc, val_precision, val_recall))
    tb_dict['eval/acc'] = val_acc
    tb_dict['eval/precision'] = val_precision
    tb_dict['eval/recall'] = val_recall
    tb_dict['eval/f1'] = val_f1
    tb_dict['eval/map'] = val_map
    tb_logger.update(tb_dict, args.train['epochs'])

    # Save final model (split module from dataparallel)
    final_model_path = os.path.join(args.log_path, 'final.pth')
    torch.save(model.module.state_dict(), final_model_path)
    logger.info(f'{final_model_path} Saved.')


def run(args):
    # Count GPUs
    n_gpus = torch.cuda.device_count()
    logger.info(f'{n_gpus} GPUs Available.')

    # Dataset
    # VOC2012
    if args.dataset == 'voc12':
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
    
    logger.info('Training Classifier...')
    # Multiprocessing (But 1 process)
    multiprocessing.spawn(_work, nprocs=1, args=(args, dataset_train, dataset_val, dataset_train_ulb), join=True)
    
    logger.info('Done Finetuning.\n')

    return None
