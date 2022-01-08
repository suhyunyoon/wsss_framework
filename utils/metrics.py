import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

def average_precision(pred, label):
    epsilon = 1e-8
    pred, label = pred.numpy(), label.numpy()
    # sort examples
    indices = pred.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(pred), 1)))

    label_ = label[indices]
    ind = label_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def AP(label, logit):
    if np.size(logit) == 0:
        return 0
    ap = np.zeros((logit.shape[1]))
    # compute average precision for each class
    for k in range(logit.shape[1]):
        # sort scores
        logits = logit[:, k]
        preds = label[:, k]
        
        ap[k] = average_precision(logits, preds)
    return ap


# for evaluation
#from sklearn.metrics import classification_report, multilabel_confusion_matrix
def eval_multilabel_metric(label, logit, average="samples"):
    pred = logit >= 0.5

    acc = accuracy_score(label, pred) * 100
    precision = precision_score(label, pred, average=average, zero_division=0) * 100
    recall = recall_score(label, pred, average=average) * 100
    f1 = f1_score(label, pred, average=average) * 100
    ap = AP(label, pred) * 100
    map = ap.mean()
    
    return acc, precision, recall, f1, ap, map
