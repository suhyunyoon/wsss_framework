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

def print_iou(iou):
    voc_class = get_voc_class()
    # miou
    miou = np.nanmean(iou)
    # print
    for k, i in zip(voc_class, iou):
        print('%-15s: %.6f' % (k,  i))
    print('%-15s: %.6f' % ('miou', np.nanmean(iou)))

# calculate iou and miou
def calc_iou(pred, seg, verbose=False):
    # calc confusion matrix
    confusion = calc_semantic_segmentation_confusion(pred, seg)

    if verbose:
        print(confusion.shape)
        
    # iou
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    # miou
    miou = np.nanmean(iou)

    if verbose:
        print_iou(iou)

    return iou, miou

def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Evaluating CAM...')
   
    # Evaluated thresholds
    eval_thres = np.arange(args.eval_thres_start, args.eval_thres_limit, args.eval_thres_jump)

    # Evaluation
    cam_list = glob.glob(os.path.join(args.cam_out_dir, '{}_*.pickle'.format(args.network)))
    #cam_path = os.path.join(args.cam_out_dir, 'cam_{}.pickle'.format(args.network))
    print(cam_list)
    
    # Read CAM
    res = {'segs': {th:[] for th in eval_thres}, 'preds': {th:[] for th in eval_thres}}
    for cam_path in cam_list:
        # Read CAM files
        print("Read CAM files...")
        with open(cam_path, 'rb') as f:
            r = pickle.load(f)
        #print(len(res['segs']), res['segs'][0].shape)
        # concat
        for th in eval_thres:
            res['segs'][th] += r['segs'][th]
            res['preds'][th] += r['preds'][th]

    # Calc ious
    ious, mious = [], []
    print("Calculate ious...")
    for th in tqdm(eval_thres):
        iou, miou = calc_iou(res['preds'][th], res['segs'][th], verbose=False)
        ious.append(iou)
        mious.append(miou)
    # Find Best thres
    best_miou = max(mious)
    best_idx = mious.index(best_miou)
    best_thres = eval_thres[best_idx]
    # Print Best mIoU
    print('Best CAM threshold: %.4f'%best_thres)
    print_iou(ious[best_idx])

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done Evaluating CAM.')
    print()

