import numpy as np
from sklearn import metrics


def get_auc(prd_array, gt_array):
    prd_array = np.array(prd_array)
    gt_array = np.array(gt_array)


    roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

    print('roc_auc: {:.4f} pr_auc: {:.4f}'.format(roc_aucs, pr_aucs))

    roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
    pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)

    return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all


print("sc")
prd_array_1 = np.load("candidate-short-chunk.npy")
gt_array_1 = np.load("candidate-ground-truth.npy")
roc_auc, pr_auc, _, _ = get_auc(prd_array_1, gt_array_1)

print("noisy student")
prd_array_4 = np.load("candidate-noisy-student.npy")
gt_array_4 = np.load("candidate-ground-truth.npy")
roc_auc, pr_auc, _, _ = get_auc(prd_array_4, gt_array_4)

print("ensemble")
prd_array_5 = np.load("candidate-ensemble.npy")
gt_array_5 = np.load("candidate-ground-truth.npy")
roc_auc, pr_auc, _, _ = get_auc(0.7 * prd_array_1 + 0.3 * prd_array_4, gt_array_1)
