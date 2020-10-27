import torch
from sklearn.metrics import *
import numpy as np
import torch.nn as nn

def feedbackloss(valid_results):
    results = list(zip(*valid_results))
    cm = confusion_matrix(results[0], results[1])
    recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    precision = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    fscore = [0, 0, 0, 0, 0]
    for cm_i in range(0, 5):
        fscore[cm_i] = 2 * recall[cm_i][cm_i] * precision[cm_i][cm_i] / (
                    recall[cm_i][cm_i] + precision[cm_i][cm_i])
    weight_new = [fscore[0], fscore[1], fscore[2], fscore[3], fscore[4]]

    weight_mean = np.mean(weight_new)
    weight_std = np.std(weight_new, ddof=1)
    for z in range(len(weight_new)):
        weight_new[z] = (weight_new[z] - weight_mean) / weight_std
        weight_new[z] = max((-1 * weight_new[z] + 1), -0.24 * weight_new[z] + 1)

    weight = torch.tensor([weight_new[0], weight_new[1], weight_new[2], weight_new[3], weight_new[4]],
                          dtype=torch.float)
    print(weight)

    FeedbackLoss = nn.CrossEntropyLoss(weight=weight)  # 可以给样本加权重，相当于惩罚机制
    return FeedbackLoss
