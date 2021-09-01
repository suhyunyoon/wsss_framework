import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from torch.backends import cudnn

#from torchvision.datasets import VOCSegmentation

import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import glob

#from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import get_model, get_cam_target_layer
from data.datasets import get_transform, VOCEvaluationCAM
from torchutils._utils import split_dataset

cudnn.enabled = True

def _work(pid, model, dataset, args):
    # split dataset
    databin = dataset[pid]
    n_gpus = torch.cuda.device_count()
    # dataloader
    dl = DataLoader(databin, shuffle=False, batch_size=1, num_workers=args.num_workers // n_gpus, pin_memory=False)

    res = {}
    with cuda.device(pid):
        model.cuda()
        
        # target layer
        target_layer = get_cam_target_layer(model, args.model_type)

        # Function which makes CAM
        make_cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=True)
        # find best CAM threshold
        eval_thres = np.arange(args.eval_thres_start, args.eval_thres_limit, args.eval_thres_jump)
        for th in tqdm(eval_thres):
            # memorize label and predictions
            segs=[]
            preds=[]
            # Generate CAM
            for img, seg in dl:
                img = img.cuda()
                # squeeze & remove border
                seg = np.squeeze(np.array(seg, dtype=np.uint8), axis=0)
                seg[seg==255] = 0
                 
                # get image classes(without background)
                label = np.unique(seg)
                label = np.intersect1d(np.arange(1, args.voc_class_num), label)
                
                # Generate CAM
                img = img.repeat(len(label),1,1,1)
                pred_cam = make_cam(input_tensor=img, target_category=label-1)
                
                # Add background
                label = np.pad(label, (1,0), mode='constant', constant_values=0)
                pred_cam = np.pad(pred_cam, ((1,0),(0,0),(0,0)), mode='constant', constant_values=th)
                pred = np.argmax(pred_cam, axis=0)
                pred = label[pred]
                
                # Append
                segs.append(seg)
                preds.append(pred)
            # Save cam by thres
            res['segs_%d' % th] = segs.copy()
            res['preds_%d' % th] = preds.copy()
    # Save CAM
    file_name = 'cam_{}.pickle'.format(pid)
    file_name = os.path.join(args.cam_out_dir, file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(res, f)

def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Generating CAM...')
    
    # Model
    weights_path = os.path.join(args.weights_dir, args.network + '.pth') 
    model, model_type = get_model(args.network, pretrained=True)
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.eval()

    args.model_type = model_type

    # GPUs
    n_gpus = torch.cuda.device_count()

    # Dataset
    transform_train = get_transform('train', args.crop_size)
    transform_target = get_transform('target', args.crop_size)
    dataset = VOCEvaluationCAM(root=args.voc12_root, year='2012', image_set='train', download=False, transform=transform_train, target_transform=transform_target)
    # Split Dataset
    dataset = split_dataset(dataset, n_gpus)
     
    # Clean directory
    file_list = glob.glob(os.path.join(args.cam_out_dir, '*'))
    for f in file_list:
        os.remove(f)
    
    # Generate CAM with Multiprocessing  
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    torch.cuda.empty_cache()

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done Generating CAM.')
    print()
