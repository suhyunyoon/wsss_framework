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

from models.utils import get_model, get_cam_target_layer
from data.datasets import get_transform, VOCEvaluationCAM
from torchutils._utils import split_dataset

cudnn.enabled = True

# reshape_transform for vits and swin
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), 
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_7(tensor, height=7, width=7):
    return reshape_transform(tensor, height, width)

def reshape_transform_14(tensor, height=14, width=14):
    return reshape_transform(tensor, height, width)

def reshape_transform_28(tensor, height=28, width=28):
    return reshape_transform(tensor, height, width)

def get_reshape_transform(model_name, model_type=''):
    if model_name in []:
        target_tr = reshape_transform_7
    elif model_name in ['dino_vits16', 'dino_vitb16', 'swin']:
        target_tr = reshape_transform_14
    elif model_name in ['dino_vits8', 'dino_vitb8']:
        target_tr = reshape_transform_28
    else:
        target_tr = None
    return target_tr

def _work(pid, dataset, args):
    # multiprocessing.spawn bug가 해결되면 run 함수로 옮기기
    # Model
    weights_path = os.path.join(args.weights_dir, args.network + '.pth') 
    model, model_type = get_model(args.network, pretrained=False, classifier=True, class_num=args.voc_class_num-1)
    
    #model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(weights_path), strict=False)
    
    #model = model.module
    model.eval() 
    args.model_type = model_type

    # split dataset
    databin = dataset[pid]
    n_gpus = torch.cuda.device_count()
    # dataloader
    dl = DataLoader(databin, shuffle=False, batch_size=1, num_workers=args.num_workers // n_gpus, pin_memory=False)
    with cuda.device(pid):
        model.cuda()
        
        # target layer
        target_layer = get_cam_target_layer(model, args.model_type)

        # Function which makes CAM
        target_tr = get_reshape_transform(args.network, model_type)
        make_cam = GradCAMPlusPlus(model=model, target_layer=target_layer, use_cuda=True, reshape_transform=target_tr)
        # save CAM per threshold
        eval_thres = np.arange(args.eval_thres_start, args.eval_thres_limit, args.eval_thres_jump)
        res = {'segs': {th:[] for th in eval_thres}, 'preds': {th:[] for th in eval_thres}}
        # Generate CAM
        for img, seg in tqdm(dl):
            img = img.cuda()
            # squeeze & remove border
            seg = np.squeeze(np.array(seg, dtype=np.uint8), axis=0)
            seg[seg==255] = 0
             
            # get image classes(without background)
            label = np.unique(seg)
            label = np.intersect1d(np.arange(1, args.voc_class_num), label)
            #print(label)

            # Generate CAM
            img = img.repeat(len(label),1,1,1)
            pred_cam = make_cam(input_tensor=img, target_category=label-1)
            #print(pred_cam.shape) 
            #import matplotlib.pyplot as plt
            #for cam in pred_cam:
            #    plt.imshow(cam)
            #    plt.show()

            # Add background
            label = np.pad(label, (1,0), mode='constant', constant_values=0)

            # Iter by thresholds
            for th in eval_thres:
                # Add background(CAM)
                cam_th = np.pad(pred_cam, ((1,0),(0,0),(0,0)), mode='constant', constant_values=th/100)
                pred = np.argmax(cam_th, axis=0)
                pred = label[pred]
                
                # Append
                res['segs'][th].append(seg)
                res['preds'][th].append(pred)

    # Save CAM
    file_name = '{}_{}.pickle'.format(args.network, pid)
    file_name = os.path.join(args.cam_out_dir, file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(res, f)
    
def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Generating CAM...')

    # GPUs
    n_gpus = torch.cuda.device_count()

    # Dataset
    transform_train = get_transform('train', args.crop_size)
    transform_target = get_transform('target', args.crop_size)
    dataset = VOCEvaluationCAM(root=args.voc12_root, year='2012', image_set=args.eval_set, download=False, transform=transform_train, target_transform=transform_target)
    # Split Dataset
    dataset = split_dataset(dataset, n_gpus)
     
    # Clean directory
    file_list = glob.glob(os.path.join(args.cam_out_dir, args.network+'*'))
    for f in file_list:
        os.remove(f)
    
    # Generate CAM with Multiprocessing  
    print('...')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(dataset, args), join=True)
    torch.cuda.empty_cache()

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done Generating CAM.')
    print()
