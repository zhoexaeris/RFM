from __future__ import print_function, division, absolute_import
import torch
import numpy as np
import random
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc, roc_auc_score

__all__ = ["data_prefetcher", "data_prefetcher_two", "cal_fam", "cal_normfam", "setup_seed", "l2_norm", "calRes"]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class data_prefetcher():
    def __init__(self, loader):
        self.stream = torch.cuda.Stream()
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class data_prefetcher_two():
    def __init__(self, loader1, loader2):
        self.stream = torch.cuda.Stream()
        self.loader1 = iter(loader1)
        self.loader2 = iter(loader2)
        self.preload()

    def preload(self):
        try:
            tmp_input1, tmp_target1 = next(self.loader1)
            tmp_input2, tmp_target2 = next(self.loader2)
            self.next_input, self.next_target = torch.cat((tmp_input1, tmp_input2)), torch.cat((tmp_target1, tmp_target2))

        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm+1e-8)
    return output


def cal_fam(model, inputs):
    model.zero_grad()
    inputs = inputs.detach().clone()
    inputs.requires_grad_()
    output = model(inputs)

    target = output[:, 1]-output[:, 0]
    target.backward(torch.ones(target.shape).cuda())
    fam = torch.abs(inputs.grad)
    fam = torch.max(fam, dim=1, keepdim=True)[0]
    return fam


def cal_normfam(model, inputs):
    fam = cal_fam(model, inputs)
    _, x, y = fam[0].shape
    fam = torch.nn.functional.interpolate(fam, (int(y/2), int(x/2)), mode='bilinear', align_corners=False)
    fam = torch.nn.functional.interpolate(fam, (y, x), mode='bilinear', align_corners=False)
    for i in range(len(fam)):
        fam[i] -= torch.min(fam[i])
        fam[i] /= torch.max(fam[i])
    return fam


def Eval(model, lossfunc, dataloader):
    model.eval()
    loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in dataloader:
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()
            outputs = model(imgs)
            loss += lossfunc(outputs, labels).item()
            y_true.append(labels)
            y_pred.append(outputs)
    
    loss = loss / len(dataloader)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    
    return loss, y_true, y_pred


def calRes(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Handle both 1D and 2D prediction arrays
    if len(y_pred.shape) == 1:
        y_pred_2d = y_pred
    else:
        y_pred_2d = y_pred[:, 1]
    
    # Calculate AP (Average Precision)
    ap = average_precision_score(y_true, y_pred_2d)
    
    # Calculate accuracy
    if len(y_pred.shape) == 1:
        y_pred_class = (y_pred_2d > 0.5).astype(int)
    else:
        y_pred_class = np.argmax(y_pred, axis=1)
    acc = np.mean(y_true == y_pred_class)
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_2d)
    
    # Calculate TPR at different FPR thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_2d)
    
    # Find TPR at FPR = 0.02, 0.03, 0.04
    tpr_2 = tpr[np.argmin(np.abs(fpr - 0.02))]
    tpr_3 = tpr[np.argmin(np.abs(fpr - 0.03))]
    tpr_4 = tpr[np.argmin(np.abs(fpr - 0.04))]
    
    return ap, acc, auc, tpr_2, tpr_3, tpr_4


def roc_curve(y_true, y_score):
    """Calculate ROC curve points"""
    # Sort scores and corresponding true values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Calculate TPR and FPR
    tpr = []
    fpr = []
    thresholds = []
    
    for threshold in np.unique(y_score):
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        thresholds.append(threshold)
    
    return np.array(fpr), np.array(tpr), np.array(thresholds)
