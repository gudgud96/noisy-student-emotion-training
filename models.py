import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
import random

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, stride=(2, 2)):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.stride = stride

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps, stride=self.stride)
        
    def gem(self, x, p=3, eps=1e-6, stride=(2, 2)):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), stride, stride=2).pow(1./p)


class CRNNBlock(nn.Module):
    def __init__(self, n_mels, d, num_class):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.gem1, self.gem2, self.gem3 = GeM(), GeM(), GeM()
        self.gru = nn.GRU(64 * (n_mels // (2**3)), d, batch_first=True, bidirectional=True)
        self.rnn = nn.GRUCell(d + num_class, d)
        self.linear = nn.Linear(512, d)

    def forward(self, x):
        identity = x
        x = self.encoder(x)       # (batch, ch, mel, time)
        x += identity
        x = self.gem1(x)
        
        identity2 = x
        x = self.enc2(x)       # (batch, ch, mel, time)
        x += identity2
        x = self.gem2(x)

        identity3 = x
        x = self.enc3(x)       # (batch, ch, mel, time)
        x += identity3
        x = self.gem3(x)

        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x = self.gru(x)[-1]
        x = torch.cat([x[0], x[1]], dim=-1)
        z = nn.ReLU()(self.linear(x))
        return z


class StochasticCRNNBLock(nn.Module):
    def __init__(self, n_mels, d, num_class, prob, n_layers):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  )
        self.max_pool_1 = nn.MaxPool2d(2, stride=2)

        self.encoders = nn.ModuleList([nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.1)) for _ in range(n_layers - 3)])

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.max_pool_3 = nn.MaxPool2d(2, stride=2)
        self.gru = nn.GRU(64 * (n_mels // (2**3)), d, batch_first=True, bidirectional=True)
        self.rnn = nn.GRUCell(d + num_class, d)
        self.linear = nn.Linear(512, d)
        self.prob = prob

    def forward(self, x):
        x = self.enc1(x)
        x = self.max_pool_1(x)

        if len(self.encoders) > 0:
            for layer in self.encoders:
                if random.random() < self.prob:
                    layer[0].weight.requires_grad = True
                    layer[1].weight.requires_grad = True
                    x = layer(x)
                else:
                    layer[0].weight.requires_grad = False
                    layer[1].weight.requires_grad = False               
             
        x = self.enc2(x)
        x = self.max_pool_2(x)
        
        x = self.enc3(x)
        x = self.max_pool_3(x)
        
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x = self.gru(x)[-1]
        x = torch.cat([x[0], x[1]], dim=-1)
        z = nn.ReLU()(self.linear(x))
        return z


class HPCPModel(nn.Module):
    def __init__(self, n_mels, n_hpcp, d, num_class):
        super().__init__()
        self.vae_mel = CRNNBlock(n_mels, d, num_class)
        self.vae_hpcp = CRNNBlock(n_hpcp, d, num_class)
        self.extractor = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, num_class),
            nn.Sigmoid(),
        )

    def forward(self, x, x_hpcp):
        z_mel = self.vae_mel(x)
        z_hpcp = self.vae_hpcp(x_hpcp)
        pred = self.extractor(torch.cat([z_mel, z_hpcp], dim=-1))
        return pred


class SingleModel(nn.Module):
    def __init__(self, n_mels, d, num_class):
        super().__init__()
        self.vae_mel = CRNNBlock(n_mels, d, num_class)
        self.extractor = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, num_class),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z_mel = self.vae_mel(x)
        pred = self.extractor(z_mel)
        return pred


class HPCPModelv2(nn.Module):
    def __init__(self, n_mels, n_hpcp, d, num_class, num_class_combined=19):
        super().__init__()
        self.vae_mel = CRNNBlock(n_mels, d, num_class)
        self.vae_hpcp = CRNNBlock(n_hpcp, d, num_class)
        self.extractor = nn.Sequential(
            nn.Linear(d * 2 + num_class_combined, d),
            nn.ReLU(),
            nn.Linear(d, num_class),
            nn.Sigmoid(),
        )
        self.combined_extractor = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, num_class_combined),
            nn.Sigmoid(),
        )

    def forward(self, x, x_hpcp):
        z_mel = self.vae_mel(x)
        z_hpcp = self.vae_hpcp(x_hpcp)
        pred_combined = self.combined_extractor(torch.cat([z_mel, z_hpcp], dim=-1))
        pred = self.extractor(torch.cat([z_mel, z_hpcp, pred_combined], dim=-1))
        return pred


class NoisyHPCPModel(nn.Module):
    def __init__(self, n_mels, n_hpcp, d, num_class, prob=1, n_layers=3):
        super().__init__()
        self.vae_mel = StochasticCRNNBLock(n_mels, d, num_class, prob=prob, n_layers=n_layers)
        self.vae_hpcp = StochasticCRNNBLock(n_hpcp, d, num_class, prob=prob, n_layers=n_layers)
        self.extractor = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(d, num_class),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )

    def forward(self, x, x_hpcp):
        z_mel = self.vae_mel(x)
        z_hpcp = self.vae_hpcp(x_hpcp)
        pred = self.extractor(torch.cat([z_mel, z_hpcp], dim=-1))
        return pred


class NoisyHPCPModelv2(nn.Module):
    def __init__(self, n_mels, n_hpcp, d, num_class, num_class_combined=19, prob=1, n_layers=3):
        super().__init__()
        self.vae_mel = StochasticCRNNBLock(n_mels, d, num_class, prob=prob, n_layers=n_layers)
        self.vae_hpcp = StochasticCRNNBLock(n_hpcp, d, num_class, prob=prob, n_layers=n_layers)
        self.extractor = nn.Sequential(
            nn.Linear(d * 2 + num_class_combined, d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d, num_class),
            nn.Sigmoid(),
        )
        self.combined_extractor = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d, num_class_combined),
            nn.Sigmoid(),
        )

    def forward(self, x, x_hpcp):
        z_mel = self.vae_mel(x)
        z_hpcp = self.vae_hpcp(x_hpcp)
        pred_combined = self.combined_extractor(torch.cat([z_mel, z_hpcp], dim=-1))
        pred = self.extractor(torch.cat([z_mel, z_hpcp, pred_combined], dim=-1))
        return pred