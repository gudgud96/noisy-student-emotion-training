import torch
from torch import nn
import numpy as np
from data_loader import get_audio_loader
import datetime
from sklearn import metrics
import time
from models import HPCPModel, NoisyHPCPModel, SingleModel
import argparse


def test(model_mode):
    print("Model:", model_mode)

    # load model
    if model_mode == "normal":
        model = SingleModel(128, 256, 56).cuda()
        model_name = "short-chunk-single-v1--roc.pth"
        
    elif model_mode == "hpcp":
        model = HPCPModel(128, 12, 256, 56).cuda()
        model_name = "short-chunk-mel-hpcp-v1.pth"

    else:
        model = NoisyHPCPModel(128, 12, 256, 56, prob=1, n_layers=6).cuda()
        model_name = "noisy-short-chunk-mel-hpcp-v1.pth"

    model.eval()
    S = torch.load(model_name)
    model.load_state_dict(S)

    # load dataset
    test_dl = get_audio_loader(root="E://mtg-jamendo-melspec-10//",
                            subset="moodtheme",
                            batch_size=1,
                            tr_val='test', 
                            split=0,
                            is_hpcp=True,
                            shuffle=False,
                            is_test_mode=True)

    # init variables
    start_t = time.time()
    prd_array = []  # prediction
    gt_array = []   # ground truth
    val_ctr = 0

    # evaluate
    for x, x_hpcp, y, _ in test_dl:
        val_ctr += 1

        # variables to cuda
        x = x.unsqueeze(1).cuda()
        x_hpcp = x_hpcp.unsqueeze(1).cuda()
        y = y.cuda()[0].unsqueeze(0)

        # predict
        with torch.no_grad():
            if model_mode == "normal":
                pred = model(x)
            else:
                pred = model(x, x_hpcp)
        
        pred = torch.mean(pred, dim=0).unsqueeze(0)
        
        loss = nn.BCELoss()(pred, y)

        # print log
        print("TEST [%s] Iter [%d/%d] valid loss: %.4f Elapsed: %s" %
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                val_ctr, len(test_dl), loss.item(),
                datetime.timedelta(seconds=time.time()-start_t)), end="\r")

        # append prediction
        pred = pred.detach().cpu()
        y = y.detach().cpu()
        for prd in pred:
            prd_array.append(list(np.array(prd)))
        for gt in y:
            gt_array.append(list(np.array(gt)))

        assert pred.shape[-1] == 56

    # get auc
    roc_auc, pr_auc, _, _ = get_auc(prd_array, gt_array)

    return roc_auc, pr_auc


def get_auc(prd_array, gt_array):
    prd_array = np.array(prd_array)
    gt_array = np.array(gt_array)

    roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

    print("")
    print('roc_auc: {:.4f} pr_auc: {:.4f}'.format(roc_aucs, pr_aucs))

    roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
    pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)

    return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="short chunk model type. choose among 'normal', 'hpcp', 'hpcp-noisy'")

    args = parser.parse_args()
    if args.model not in ['normal', 'hpcp', 'hpcp-noisy']:
        print("No specified model model. Ending...")
    else:
        roc_auc, _ = test(args.model)
