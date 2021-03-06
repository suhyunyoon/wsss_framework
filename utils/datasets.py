import torch
from torchvision.datasets import VOCSegmentation, VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
# from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, ToTensor

import numpy as np
import os


# ImageNet
# !wget https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt
def get_imagenet_class(src='./data/imagenet.txt'):
    with open(src, 'r') as f:
        #lines = f.readlines()
        #lines = list(map(lambda x:x.split(':'), lines))
        #imagenet_class = {int(k.strip()): v.strip()[1:-2] for k, v in lines}
        
        lines = f.read()
        lines = list(map(lambda x:x.split(':')[1], lines[1:].split('\n')[:-1]))
    imagenet_class = [line.strip()[1:-2] for line in lines]

    return imagenet_class


# VOC
# VOC class names
voc_class = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
voc_class_num = len(voc_class)

# VOC COLOR MAP
voc_colormap = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

def get_voc_class():
    return voc_class

def get_voc_colormap():
    return voc_colormap


# transformation
voc_mean = [0.485, 0.456, 0.406]
voc_std = [0.229, 0.224, 0.225]
#h,w = 520, 520
#h,w = 256, 256 -> RandomCrop 224

def voc_train_dataset(args, img_list, mode='cls'):
    tfs_train = tfs.Compose([tfs.Resize((args.train['input_size'], args.train['input_size'])),  
                            tfs.RandomHorizontalFlip(),
                            tfs.RandomCrop(args.train['crop_size'], padding=4, padding_mode='reflect'),
                            #tfs.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                            tfs.ToTensor(),
                            tfs.Normalize(voc_mean, voc_std),
                            ])
    tfs_target = tfs.Compose([tfs.Resize((args.train['crop_size'], args.train['crop_size']))
                            ])

    if mode == 'cls':
        dataset = VOCClassification(root=args.dataset_root, year='2012', image_set='train', 
                                    dataset_list=img_list, download=False, transform=tfs_train)
    elif mode == 'seg':
        dataset = VOCSegmentationInt(root=args.dataset_root, year='2012', image_set='train', 
                                 download=False, transform=tfs_train, target_transform=tfs_target)

    return dataset

def voc_val_dataset(args, img_list, mode='cls'):
    tfs_val = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size'])),  
                            tfs.ToTensor(),
                            tfs.Normalize(voc_mean, voc_std),
                            ])
    tfs_target = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size']))
                            ])

    if mode == 'cls':
        dataset = VOCClassification(root=args.dataset_root, year='2012', image_set='val', 
                                    dataset_list=img_list, download=False, transform=tfs_val)
    elif mode == 'seg':
        dataset = VOCSegmentationInt(root=args.dataset_root, year='2012', image_set='val',
                                 download=False, transform=tfs_val, target_transform=tfs_target)
    return dataset

def voc_test_dataset(args, img_list, mode='cls'):
    tfs_test = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size'])),
                            tfs.ToTensor(),
                            tfs.Normalize(voc_mean, voc_std),
                            ])
    tfs_target = tfs.Compose([tfs.Resize((args.eval['crop_size'], args.eval['crop_size']))
                            ])

    if mode == 'cls':
        dataset = VOCClassification(root=args.dataset_root, year='2012', image_set='test', 
                                    dataset_list=img_list, download=False, transform=tfs_test)
    elif mode == 'seg':
        dataset = VOCSegmentationInt(root=args.dataset_root, year='2012', image_set='test', 
                                 download=False, transform=tfs_test, target_transform=tfs_target)
    return dataset
    

def re_normalize(x, mean=voc_mean, std=voc_std):
    x_r = x.clone()
    for c, (mean_c, std_c) in enumerate(zip(mean,std)):
        x_r[c] *= std_c
        x_r[c] += mean_c
    return x_r

class VOCSegmentationInt(VOCSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, seg = super().__getitem__(index)
        #seg = torch.LongTensor(seg)
        seg = np.array(seg, dtype=np.uint8)
        return img, seg


class VOCClassification(VOCDetection):
    def __init__(self, *args, **kwargs):
        # Init
        self.dataset_list = kwargs['dataset_list']
        kwargs.pop('dataset_list', None)

        super(VOCClassification, self).__init__(*args, **kwargs)

        self.voc_class = voc_class
        self.voc_class_num = voc_class_num
        self.voc_colormap = voc_colormap
        
        # directory initialization
        image_dir = os.path.split(self.images[0])[0]
        annotation_dir = os.path.join(os.path.dirname(image_dir), 'Annotations')

        # read list of train_aug
        with open(self.dataset_list, 'r') as f:
            train_aug = f.read().split()
        # replace train into train_aug(images, annotations)
        self.images = [os.path.join(image_dir, x + ".jpg") for x in train_aug]
        
        # deprecated(read-only property)
        #self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in train_aug]
        # Re-append xml file list
        self.annotations.clear()
        for x in train_aug:
            self.annotations.append(os.path.join(annotation_dir, x + ".xml"))

    def __getitem__(self, index):
        img, ann = super().__getitem__(index)
        
        # get object list
        objects = ann['annotation']['object']
        # get unique classes
        ann = torch.LongTensor(list({self.voc_class.index(o['name'])-1 for o in objects}))
        # make one-hot encoding
        one_hot = torch.zeros(self.voc_class_num-1)
        one_hot[ann] = 1

        return img, one_hot
