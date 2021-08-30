from chainercv.evaluations import calc_semantic_segmentation_confusion
from data.classes import get_voc_class

def calc_iou(pred, seg, verbose=False):
    voc_class = get_voc_class()

    confusion = calc_semantic_segmentation_confusion(preds, segs)

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
        for k, i in zip(voc_class, iou):
            print('%-15s:' % k,  i)
        print('%-15s:' % 'miou', np.nanmean(iou))

    return iou, miou

