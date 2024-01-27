import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import torch
from torch import nn
from sklearn.metrics import classification_report,\
    roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import h5py

TEST_SIZE = .1

# model definition
mult = 8
class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(6, mult*256),
          nn.LeakyReLU(),
          nn.Linear(mult*256, mult*256),
          nn.Dropout(.1),
          nn.LeakyReLU(),
          nn.Linear(mult*256, mult*512),
          nn.Dropout(.1),
          nn.LeakyReLU(),
          nn.Linear(mult*512, 1501),
        )
    self.loss = nn.MSELoss()
    
    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch    
        x = x.float()
        y = y.float()
        perm = torch.randperm(3,device='cuda')
        perm = torch.cat((perm, perm+3))
        x = torch.index_select(x, 1, perm)

        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        y_hat = torch.clamp(y_hat, 0)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# data
df = pd.read_csv('../processed_data/6d_stability.csv')
f = h5py.File('../processed_data/6d_dos.hdf5', 'r')

# cleaning
sums = []
for x in tqdm.tqdm(f['dos']):
    sums.append(sum(x))
bad_samples = np.nonzero(np.array(sums) < 50)[0]  # calculation artefacts
idx = list(np.nonzero(df.is_stable.values > 0)[0])
idx = [x for x in idx if x not in bad_samples]
train_size = int(len(idx)*(1 - TEST_SIZE))
idx = np.random.permutation(idx)
train_idx = idx[:train_size]
test_idx = idx[train_size:]
X_train = df[['exx', 'eyy', 'ezz', 'eyz', 'exz', 'exy']].values[train_idx]
X_test = df[['exx', 'eyy', 'ezz', 'eyz', 'exz', 'exy']].values[test_idx]
y_train = np.array(f['dos'])[train_idx]
y_test  = np.array(f['dos'])[test_idx]
X_tr_tensor, y_tr_tensor = torch.tensor(X_train,device='cuda'), \
                            torch.tensor(y_train,device='cuda')

# model 
mlp = MLP()
train_dset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_dloader = DataLoader(train_dset, batch_size=8000)
trainer = pl.Trainer(gpus=1,
                     auto_lr_find=True,
                     max_epochs=20000)
trainer.fit(mlp, train_dloader)
trainer.save_checkpoint("./saved_models/dos_6d.ckpt")

# assess accuracy
tst = torch.tensor(X_test,device='cuda',dtype=torch.float)
pred = mlp(tst).detach().cpu().numpy()
pred = np.clip(pred, 0, None)  # important as DOS is non-negative
maes = []
for q in range(len(pred)):
    maes.append(np.abs(pred[q]-y_test[q]).mean())
print(np.mean(maes))
