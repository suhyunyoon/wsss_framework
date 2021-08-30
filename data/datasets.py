import torch
from torchvision.datasets import VOCSegmentation, VOCDetection
from torch.utils.data import DataLoader

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

# transformation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
#h,w = 520, 520
#h,w = 256, 256 -> RandomCrop 224

def get_transform(split, hw):
    transform = None
    h, w = hw, hw
    if split == 'train':
        transform = Compose([Resize((h,w)),
                                   ToTensor(),
                                    #RandomHorizontalFlip(p=0.5),
                                    Normalize(mean, std)])
    elif split == 'val':
        transform = Compose([Resize((h,w)),
                                ToTensor(),
                                Normalize(mean, std)])
    elif split == 'target':
        transform = Compose([Resize((h,w))])
                                   #ToTensor()])
    return transform

class VOCClassification(VOCDetection):
    def __init__(self, root='/home/suhyun/dataset/VOC/', year='2012', image_set='train', download=False, transform=None):
        super().__init__(root=root, year=year, image_set=image_set, download=download, transform=transform)
        self.voc_class = voc_class
        self.voc_class_num = voc_class_num
        self.voc_colormap = voc_colormap

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
