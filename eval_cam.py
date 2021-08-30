import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from torch.backends import cudnn

from torchvision.datasets import VOCSegmentation

import os
import numpy as np
from datetime import datetime

#from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import get_model
from data.dataset import get_transform

cudnn.enabled = True

def _work(pid, model, dataset, args):
    # split dataset
    databin = dataset[pid]
    n_gpus = torch.cuda.device_count()
    # dataloader
    dl = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(pid):
        model.cuda()
        
        #
        target_layer = None

        # Function which makes CAM
        make_cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=True)
        
        # memorize label and predictions
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
            pred_cam = np.pad(pred_cam, ((1,0),(0,0),(0,0)), mode='constant', constant_values=args.eval_thres)
            pred = np.argmax(pred_cam, axis=0)
            pred = label[pred]
            
            # Append
            segs.append(seg)
            preds.append(pred)


def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Evaluating CAM...')
    
    # Model
    weights_path = os.path.join(args.weights_dir, args.network + '.pth') 
    model, model_type = get_model(args.network, pretrained=False)
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.eval()

    # GPUs
    n_gpus = torch.cuda.device_count()

    # Dataset
    transform_train = get_transform('train', args.crop_size)
    transform_target = get_transform('target', args.crop_size)
    dataset = VOCSegmentation(root=args.voc12_root, year='2012', image_set='train', downld=False, transform=transform_train, target_transform=transform_target)
    # Split Dataset
    dataset = torchutils.split_dataset(dataset, n_gpus)

    # Multiprocessing
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    torch.cuda.empty_cache()

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Evaluating CAM...')

