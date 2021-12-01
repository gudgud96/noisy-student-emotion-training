import torch
from torch import nn
from models import NoisyHPCPModel, HPCPModelv2
import numpy as np
from data_loader import get_audio_loader, get_audio_loader_augment
import datetime
from sklearn import metrics
from torch.optim.lr_scheduler import ExponentialLR
import time

model = NoisyHPCPModel(128, 12, 256, 56, 19, prob=0.95, n_layers=7).cuda()
model.train()

pretrained_model = HPCPModelv2(128, 12, 256, 56).cuda()
pretrained_model.eval()
pretrained_model_name = "melspec-10-normed-randaugment-v6-gem-combined-tag-pr.pth"
S = torch.load(pretrained_model_name)
pretrained_model.load_state_dict(S)

UPPER_THRESHOLD = 0.1
LOWER_THRESHOLD = 1E-6

path = 'tag_list.npy'
tag_list = np.load(path)
tag_list = tag_list[127:]
config_subset = "moodtheme"
config_split = "0"
roc_auc_fn = 'roc_auc_'+config_subset+'_'+config_split+'.npy'
pr_auc_fn = 'pr_auc_'+config_subset+'_'+config_split+'.npy'


train_dl = get_audio_loader_augment(root="E://mtg-jamendo-melspec-10//",
                                    subset="moodtheme",
                                    batch_size=16,
                                    tr_val='train', 
                                    split=0)
val_dl = get_audio_loader(root="E://mtg-jamendo-melspec-10//",
                          subset="moodtheme",
                          batch_size=16,
                          tr_val='validation', 
                          split=0)

print('train_dl: {} val_dl: {} '.format(len(train_dl), len(val_dl)))

full_train_dl = get_audio_loader_augment(root="E://mtg-jamendo-melspec-10//",
                                         subset="all",
                                         batch_size=16,
                                         tr_val='train', 
                                         split=0)


print('full_train_dl: {}'.format(len(full_train_dl)))

n_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
scheduler = ExponentialLR(optimizer, gamma=0.9)
ctr = 0
ctr2 = 0
best_roc_auc = 0
best_pr_auc = 0
best_roc_auc_2 = 0
best_pr_auc_2 = 0
start_t = time.time()


def validation(start_t, epoch):
    prd_array = []  # prediction
    gt_array = []   # ground truth
    val_ctr = 0
    model.eval()
    for x, x_hpcp, y, _ in val_dl:
        val_ctr += 1

        # variables to cuda
        x = x.unsqueeze(1).cuda()
        x_hpcp = x_hpcp.unsqueeze(1).cuda()
        y = y.cuda()

        # predict
        with torch.no_grad():
            pred = model(x, x_hpcp)
        
        loss = nn.BCELoss()(pred, y)

        # print log
        print("VALID [%s] Epoch [%d/%d], Iter [%d/%d] pred loss: %.4f Elapsed: %s" %
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch+1, n_epochs, val_ctr, len(val_dl), loss.item(),
                datetime.timedelta(seconds=time.time()-start_t)), end="\r")

        # append prediction
        pred = pred.detach().cpu()
        y = y.detach().cpu()
        for prd in pred:
            prd_array.append(list(np.array(prd)))
        for gt in y:
            gt_array.append(list(np.array(gt)))

    # get auc
    roc_auc, pr_auc, _, _ = get_auc(prd_array, gt_array)
    return roc_auc, pr_auc


def get_auc(prd_array, gt_array):
    prd_array = np.array(prd_array)
    gt_array = np.array(gt_array)

    roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

    roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
    pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)
    return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all


for epoch in range(n_epochs):
    # labelled train
    for x, x_hpcp, y, _ in train_dl:
        ctr += 1

        t1 = time.time()
        x = x.unsqueeze(1).cuda()
        x_hpcp = x_hpcp.unsqueeze(1).cuda()
        y = y.cuda()

        # predict
        pred, pred_combined = model(x, x_hpcp)
        loss = nn.BCELoss()(pred, y)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print log
        print("TRAIN LABEL [%s] Epoch [%d/%d] Iter [%d/%d] pred: %.4f Elapsed: %s" %
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch+1, n_epochs, ctr, len(train_dl) * n_epochs, 
                loss.item(), datetime.timedelta(seconds=time.time()-start_t)), end="\r")

    # validation
    roc_auc, pr_auc = validation(start_t, epoch)
    print('roc_auc: {:.4} pr_auc: {:.4}'.format(roc_auc, pr_auc))

    # unlabelled train
    for x, x_hpcp,  _, _ in full_train_dl:
        ctr2 += 1
        # variables to cuda
        x = x.unsqueeze(1).cuda()
        x_hpcp = x_hpcp.unsqueeze(1).cuda()

        # get label
        y_est = pretrained_model(x, x_hpcp)
        y_est = y_est.detach()
    
        # use hard label
        y_est[y_est >= UPPER_THRESHOLD] = 1
        y_est[y_est <= LOWER_THRESHOLD] = 0
        
        # only mask confident positive labels
        mask = y_est.clone()
        mask[mask >= UPPER_THRESHOLD] = 1
        mask[mask <= LOWER_THRESHOLD] = 1
        mask[mask != 1] = 0

        if mask.sum() < 16:     # batch size
            continue

        # student model forward pass
        pred = model(x, x_hpcp)
        loss = nn.BCELoss()(pred, y_est)
        loss = loss * mask
        loss = loss.sum() / mask.sum()

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print log
        print("TRAIN UNLAB [%s] Epoch [%d/%d] Iter [%d/%d] pred: %.4f Elapsed: %s" %
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch+1, n_epochs, ctr2, len(full_train_dl) * n_epochs, 
                loss.item(), datetime.timedelta(seconds=time.time()-start_t)), end="\r")
        
    # validation
    roc_auc, pr_auc = validation(start_t, epoch)
    print("")
    print('roc_auc: {:.4} pr_auc: {:.4}'.format(roc_auc, pr_auc))

    # save model
    if roc_auc > best_roc_auc:
        print('best model roc_auc: {}'.format(roc_auc))
        best_roc_auc = roc_auc
        torch.save(model.state_dict(), 'your-noisy-student-model.pth')
        
    print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    scheduler.step()
    model.train()