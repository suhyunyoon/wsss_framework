import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from tqdm import tqdm

# Validation in training
def validate(model, dl, dataset, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = 0.
        logits = []
        labels = []
        for img, label in tqdm(dl):
            # memorize labels
            labels.append(label)
            img, label = img.cuda(), label.cuda()
            
            # calc loss
            logit = model(img)
            loss = criterion(logit, label).mean()
            
            # loss
            val_loss += loss.detach().cpu()
            # acc
            logit = torch.sigmoid(logit).detach()
            logits.append(logit)
        # Eval
        # loss
        val_loss /= len(dataset)
        # eval
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0)
        acc, precision, recall, ap, map = eval_multilabel_metric(labels, logits, average='samples')

    return val_loss, acc, precision, recall, ap, map

    
# Metrics
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
    ap = AP(label, pred) * 100
    map = ap.mean()
    
    return acc, precision, recall, ap, map
