# Generate Pseudo-labels from Trained Classification Model

import os
import pickle
from tqdm import tqdm

import torch

from data.classes import get_voc_class

from data.datasets import voc_val_dataset
from torch.utils.data import DataLoader

from utils.models import get_model

import logging
logger = logging.getLogger('main')

# Multi-label class prediction
def predict(model, dl, to_numpy=True):
    model.eval()
    with torch.no_grad():
        logits = []
        for img, _ in tqdm(dl):
            # memorize labels
            img = img.cuda()
            # calc loss
            logit = model(img)
            # acc
            logit = torch.sigmoid(logit).detach()
            logits.append(logit)
        # Eval
        # eval
        logits = torch.cat(logits, dim=0)
        pred = (torch.sigmoid(logits) >= 0.5).detach().cpu()

        if to_numpy:
            pred = pred.numpy()

    return pred

def run(args):
    logger.info('Generating Pseudo Labels...')

    # Log path
    if args.weights_name is None:
        weights_path = os.path.join(args.log_path, 'final.pth')
    else:
        weights_path = args.weights_name

    # Dataset
    # VOC2012
    if args.dataset == 'voc12':
        voc_class = get_voc_class()
        voc_class_num = len(voc_class)

        # Unlabeled dataset
        if args.labeled_ratio < 1.0 or args.train_ulb_list:
            dataset = voc_val_dataset(args, args.train_ulb_list, 'cls')
        # whole dataset
        else:
            dataset = voc_val_dataset(args, args.train_list, 'cls')

    logger.info(f'Dataset Length: {len(dataset)}')
    
    # Dataloader
    dl = DataLoader(dataset, batch_size=args.eval['batch_size'], num_workers=args.num_workers, 
                        shuffle=False, sampler=None, pin_memory=True)

    # Get Model
    model = get_model(args.network, pretrained=True, num_classes=voc_class_num-1)
    
    model.load_state_dict(torch.load(weights_path), strict=False)
    #model = model.module
    model.eval()
    model = model.cuda()
    
    pred = predict(model, dl, to_numpy=True)
    idx = {os.path.splitext(os.path.basename(img))[0] : i for i, img in enumerate(dataset.images)}
    pack = {'idx': idx, 'pred': pred}

    prediction_path = os.path.join(args.log_path, 'cls_pred.npy')
    with open(prediction_path, 'wb') as f:
        pickle.dump(pack, f)
        
    logger.info(f'{prediction_path} Saved.')

    return None
