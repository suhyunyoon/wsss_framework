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
#from pytorch_grad_cam.utils.image import show_cam_on_image

from utils.models import get_model, get_cam_target_layer, get_reshape_transform
from data.datasets import get_transform, VOCSegmentationInt

cudnn.enabled = True

def _work(pid, dataset, args, cfg):
    # multiprocessing.spawn bug가 해결되면 run 함수로 옮기기
    
    # Find newest default model name
    if args.weights_name is None:
        model_names = glob.glob(os.path.join(args.weights_dir, cfg['network'] + '*'))
        weights_path = max(model_names, key=os.path.getctime)
    else:
        weights_path = os.path.join(args.weights_dir, args.weights_name) 
    # Load model
    model = get_model(cfg['network'], pretrained=False, num_classes=args.voc_class_num-1)
    
    #model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(weights_path), strict=False)
    #model = model.module

    model.eval()

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
        model.cuda()
        
        # target layer
        target_layer = get_cam_target_layer(args, model)

        # Function which makes CAM
        target_tr = get_reshape_transform(cfg['network'])
        make_cam = CAM(model=model, target_layers=[target_layer], use_cuda=True, reshape_transform=target_tr)
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
    file_name = '{}_{}.pickle'.format(cfg['network'], pid)
    file_name = os.path.join(args.cam_out_dir, file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(res, f)

    # clear gpu cache
    #torch.cuda.empty_cache()
    
def run(args, cfg):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Generating CAM...')

    # GPUs
    n_gpus = torch.cuda.device_count()

    # Dataset
    transform_train = get_transform('train', cfg['model']['crop_size'])
    transform_target = get_transform('target', cfg['model']['crop_size'])
    dataset = VOCSegmentationInt(root=args.voc12_root, year='2012', image_set=args.eval_set, 
                                 download=False, transform=transform_train, target_transform=transform_target)
    # Split Dataset
    dataset = [Subset(dataset, np.arange(i, len(dataset), n_gpus)) for i in range(n_gpus)]
     
    # Clean directory
    file_list = glob.glob(os.path.join(args.cam_out_dir, cfg['network']+'*'))
    for f in file_list:
        os.remove(f)
    
    # Generate CAM with Multiprocessing  
    print('...')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(dataset, args, cfg), join=True)
    torch.cuda.empty_cache()
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done Generating CAM.')
    print()
