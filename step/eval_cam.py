# Use no GPUs
import torch
import os
import numpy as np
from datetime import datetime
import pickle
import glob
from tqdm import tqdm

from chainercv.evaluations import calc_semantic_segmentation_confusion
from data.classes import get_voc_class

import logging
logger = logging.getLogger('main')

def print_iou(iou):
    voc_class = get_voc_class()
    # miou
    miou = np.nanmean(iou)
    # print
    ret = '\n'
    for k, i in zip(voc_class, iou):
        ret += '%-15s: %.6f\n' % (k,  i)
    ret += '%-15s: %.6f' % ('miou', miou)

    logger.info(ret)


# calculate iou and miou
def calc_iou(pred, seg):
    # calc confusion matrix
    confusion = calc_semantic_segmentation_confusion(pred, seg)

    # iou
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    # miou
    miou = np.nanmean(iou)

    return iou, miou

def run(args):
    logger.info('Evaluating CAM...')
   
    # set CAM directory path
    args.cam_dir = os.path.join(args.log_path, 'cam')

    # stored CAM file list
    cam_list = glob.glob(os.path.join(args.cam_dir, '*.pickle'))
    logger.info(cam_list)

    # Evaluated thresholds
    eval_thres = np.arange(args.eval_thres_start, args.eval_thres_limit, args.eval_thres_jump)

    # Read CAM
    res = {'segs': {th:[] for th in eval_thres}, 'preds': {th:[] for th in eval_thres}}
    for cam_path in cam_list:
        # Read CAM files
        logger.info(f"Read CAM files... ({cam_path})")
        with open(cam_path, 'rb') as f:
            r = pickle.load(f)
        #print(len(res['segs']), res['segs'][0].shape)
        # concat
        for th in eval_thres:
            res['segs'][th] += r['segs'][th]
            res['preds'][th] += r['preds'][th]

    # Calc ious
    ious, mious = [], []
    logger.info("Calculate ious...")

    for th in tqdm(eval_thres):
        iou, miou = calc_iou(res['preds'][th], res['segs'][th])
        ious.append(iou)
        mious.append(miou)
        
    # Find Best thres
    best_miou = max(mious)
    best_idx = mious.index(best_miou)
    best_thres = eval_thres[best_idx]

    # Print Best mIoU
    logger.info('Best CAM threshold: %.4f'%best_thres)
    print_iou(ious[best_idx])

    logger.info('Done Evaluating CAM.\n')

