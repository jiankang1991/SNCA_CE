
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class KNNClassification(nn.Module):

    def __init__(self, X_train, Y_true, K=10):
        super().__init__()

        self.K = K

        self.KNN = KNeighborsClassifier(n_neighbors=self.K, weights='distance')
        self.KNN.fit(X_train, Y_true)

    def forward(self, X_test, y_true):

        y_pred = self.KNN.predict(X_test)

        acc = accuracy_score(y_true, y_pred)

        return acc


class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
    def forward(self, predict_labels, true_labels):

        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()



class HingeLoss(nn.Module):
    """
    Hinge loss based on the paper:
    when deep learning meets metric learning:remote sensing image scene classification
    via learning discriminative CNNs 
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/9
    """

    def __init__(self, margin=0.44):
        super().__init__()
        
        self.margin = margin

    def forward(self, oneHotCodes, features):
        
        L_S = oneHotCodes.mm(torch.t(oneHotCodes))
        Dist = torch.norm(features[:,None] - features, dim=2, p=2)**2

        Dist = self.margin - Dist
        
        L_S[L_S==0] = -1

        Dist = 0.05 - L_S * Dist

        loss = torch.triu(Dist, diagonal=1)

        loss[loss < 0] = 0

        return torch.mean(loss)












