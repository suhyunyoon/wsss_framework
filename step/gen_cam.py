# Generating with trained Model

import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader, Subset
from torch.backends import cudnn

#from torchvision.datasets import VOCSegmentation

import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import glob

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, ScoreCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget, ClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image

from utils.models import get_model, get_cam_target_layer, get_reshape_transform
from utils.datasets import voc_train_dataset, voc_val_dataset
from utils.misc import make_logger

import logging
logger = logging.getLogger('main')

cudnn.enabled = True

def _work(pid, dataset, args):
    logger, _ = make_logger(args, is_new=False)

    # Log path
    if args.weights_name is None:
        weights_path = os.path.join(args.log_path, 'final.pth')
    else:
        weights_path = args.weights_name

    # Load model
    model = get_model(args.network, pretrained=False, num_classes=args.voc_class_num-1)
    
    # Select CAM
    if args.cam_type == 'gradcam':
        CAM = GradCAM
    elif args.cam_type == 'gradcamplusplus' or args.cam_type == 'gradcam++':
        CAM = GradCAMPlusPlus
    elif args.cam_type == 'layercam':
        CAM = LayerCAM
    elif args.cam_type == 'scorecam':
        CAM = ScoreCAM
    elif args.cam_type == 'ablationCAM':
        CAM = AblationCAM
    elif args.cam_type == 'xgradcam':
        CAM = XGradCAM
    elif args.cam_type == 'eigencam':
        CAM = EigenCAM
    elif args.cam_type == 'eigengradcam':
        CAM = EigenGradCAM
    elif args.cam_type == 'fullgrad':
        CAM = FullGrad

    # split dataset
    databin = dataset[pid]
    n_gpus = torch.cuda.device_count()
    # dataloader
    dl = DataLoader(databin, shuffle=False, batch_size=1, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with cuda.device(pid):
        #model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(weights_path), strict=False)
        #model = model.module
        model.eval()
        model = model.cuda()
        
        # target layer
        target_layer = get_cam_target_layer(model)

        # Function which makes CAM
        target_tr = get_reshape_transform(args.network)
        make_cam = CAM(model=model, target_layers=[target_layer], use_cuda=True, reshape_transform=target_tr)
        # save CAM per threshold
        eval_thres = np.arange(args.eval_thres_start, args.eval_thres_limit, args.eval_thres_jump)
        # Generate CAM
        for idx, (img, seg) in enumerate(tqdm(dl)):
            img = img.cuda()
            # squeeze & remove border
            seg = np.squeeze(np.array(seg, dtype=np.uint8), axis=0)
            seg[seg==255] = 0
             
            # get image classes(without background)
            label = np.unique(seg)
            label = np.intersect1d(np.arange(1, args.voc_class_num), label)
            targets = [ClassifierOutputTarget(i-1) for i in label]
            #print(torch.sigmoid(model(img)), label)
    
            # Make CAM
            img = img.repeat(len(label),1,1,1)
            pred_cam = make_cam(input_tensor=img, targets=targets)
            
            # Add background
            label = np.pad(label, (1,0), mode='constant', constant_values=0)
    
            # Iter by thresholds
            res = {'segs': {th:[] for th in eval_thres}, 'preds': {th:[] for th in eval_thres}}
            for th in eval_thres:
                # Add background(CAM)
                cam_th = np.pad(pred_cam, ((1,0),(0,0),(0,0)), mode='constant', constant_values=th/100)
                pred = np.argmax(cam_th, axis=0)
                pred = label[pred]
                
                # Append
                res['segs'][th] = seg
                res['preds'][th] = pred
            
            file_name = os.path.join(args.cam_dir, f'{pid}_{idx}.npy') 
            np.save(file_name, res)

    # clear gpu cache
    torch.cuda.empty_cache()

def run(args):
    logger.info('Generating CAM...')
    
    # GPUs
    n_gpus = torch.cuda.device_count()

    # Dataset
    dataset = voc_val_dataset(args, args.eval_list, 'seg')
    # Split Dataset
    dataset = [Subset(dataset, np.arange(i, len(dataset), n_gpus)) for i in range(n_gpus)]

    # set Cam directory path
    args.cam_dir = os.path.join(args.log_path, 'cam')

    # Make dir
    if not os.path.exists(args.cam_dir):
        os.mkdir(args.cam_dir)
    
    # Clean existing CAMs
    file_list = glob.glob(os.path.join(args.cam_dir, '*'))
    for f in file_list:
        os.remove(f)
    
    # Generate CAM with Multiprocessing  
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(dataset, args), join=True)
    #torch.cuda.empty_cache()
    
    logger.info('Done Generating CAM.\n')
