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
def get_voc_class():
    return voc_class

def get_voc_colormap():
    return voc_colormap
