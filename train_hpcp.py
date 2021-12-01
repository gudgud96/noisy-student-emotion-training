import torch
from torch import nn
from models import HPCPModel
import numpy as np
from data_loader import get_audio_loader, get_audio_loader_augment
import datetime
from sklearn import metrics
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = HPCPModel(128, 12, 256, 56).cuda()

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
                            split=0,
                            is_hpcp=True)
val_dl = get_audio_loader(root="E://mtg-jamendo-melspec-10//",
                          subset="moodtheme",
                          batch_size=16,
                          tr_val='validation', 
                          split=0,
                          is_hpcp=True)

# pre-train an oracle
n_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
ctr = 0
best_roc_auc = 0
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

    print("")
    print('roc_auc: {:.4f} pr_auc: {:.4f}'.format(roc_aucs, pr_aucs))

    roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
    pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)
    return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all


for epoch in range(n_epochs):
    for x, x_hpcp, y, _ in train_dl:
        ctr += 1

        t1 = time.time()
        # variables to cuda
        x = x.unsqueeze(1).cuda()
        x_hpcp = x_hpcp.unsqueeze(1).cuda()
        y = y.cuda()

        # predict
        pred = model(x, x_hpcp)
        loss = nn.BCELoss()(pred, y)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print log
        print("TRAIN [%s] Epoch [%d/%d] Iter [%d/%d] pred: %.4f Elapsed: %s" %
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch+1, n_epochs, ctr, len(train_dl) * n_epochs, 
                loss.item(),
                datetime.timedelta(seconds=time.time()-start_t)), end="\r")

    # validation
    roc_auc, _ = validation(start_t, epoch)

    # save model
    if roc_auc > best_roc_auc:
        print('best model roc_auc: {}'.format(roc_auc))
        best_roc_auc = roc_auc
        torch.save(model.state_dict(), 'your-hpcp-model.pth')

    scheduler.step(roc_auc)
    
    model.train()

    print("[%s] Train finished. Elapsed: %s"
            % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                datetime.timedelta(seconds=time.time() - start_t)))