import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import torch
from torch import nn
import h5py
import matplotlib.pyplot as plt

TEST_SIZE = .1

# model definition
mult = 4
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
          nn.Linear(mult*512, 20250),
        )
        self.loss = nn.MSELoss()
    
    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch    
        x = x.float()
        y = y.float()

        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# data filtering 
df = pd.read_csv('../processed_data/6d_stability.csv')
f = h5py.File('../processed_data/6d_bandstructure.hdf5', 'r')
fx = h5py.File('../processed_data/6d_dos.hdf5', 'r')
sums = []
for x in tqdm.tqdm(fx['dos']):
    sums.append(sum(x))
fx.close()

bad_samples = np.nonzero(np.array(sums) < 50)[0]  # calculation artefacts
idx = list(np.nonzero(df.is_stable.values > 0)[0])
idx = [x for x in idx if x not in bad_samples]
train_size = int(len(idx)*(1 - TEST_SIZE))
train_idx = idx[:train_size]
test_idx = idx[train_size:]
X_train = df[['exx', 'eyy', 'ezz', 'eyz', 'exz', 'exy']].values[train_idx]
X_test = df[['exx', 'eyy', 'ezz', 'eyz', 'exz', 'exy']].values[test_idx]
y_train = np.array(f['bandstructure'][train_idx])
y_test  = np.array(f['bandstructure'][test_idx])
y_train = y_train.reshape((y_train.shape[0], -1))
y_test = y_test.reshape((y_test.shape[0], -1))
X_tr_tensor, y_tr_tensor = torch.tensor(X_train,device='cuda'), \
                            torch.tensor(y_train,device='cuda')
f.close()
train_dset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_dloader = DataLoader(train_dset, batch_size=512)

# model
mlp = MLP()
trainer = pl.Trainer(gpus=1,
                     auto_lr_find=True,
                     max_epochs=30000)
trainer.fit(mlp, train_dloader)
trainer.save_checkpoint("./saved_models/bs_6d_best.ckpt")

# accuracy
pred = mlp(torch.Tensor(X_test)).detach().numpy()
bs = pred.reshape(pred.shape[0],3375,6)
tbs = y_test.reshape(pred.shape[0],3375,6)
for band in range(6):
    maes = np.mean(np.abs(bs[:,band] - tbs[:,band]), axis = 1)\
        /np.mean(np.abs(tbs[:,band]))
    print(f"band={band}, rmae={(100*np.mean(maes)):.2f}%")