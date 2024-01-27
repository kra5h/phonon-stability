import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from torch import nn
from sklearn.metrics import classification_report,\
    roc_auc_score, f1_score

# model definition
class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        mult = 64
        self.layers = nn.Sequential(
          nn.Linear(6, 4*mult),
          nn.LeakyReLU(),
          nn.Linear(4*mult, 2*mult),
          nn.LeakyReLU(),
            nn.Linear(2*mult, mult),
          nn.LeakyReLU(),
          nn.Linear(mult, 1),
          nn.Sigmoid()
        )
        self.ce = nn.BCELoss()
    
    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch    
        x = x.float()
        y = y.float()
        perm = torch.randperm(3)#,device='cuda')
        perm = torch.cat((perm, perm+3))
        x = torch.index_select(x, 1, perm)

        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('train_loss', loss)
        return loss  
  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# data preparation
df = pd.read_csv('../processed_data/6d_stability.csv')
df['stable'] = df['is_stable'] == 1
X_train, X_test, y_train, y_test = train_test_split(
    df[['exx', 'eyy', 'ezz', 'eyz', 'exz', 'exy']].values,
    df['stable'].values,
    test_size=0.05)
y_train = y_train[:, None]
y_test = y_test[:, None]
X_tr_tensor, y_tr_tensor = torch.Tensor(X_train), torch.Tensor(y_train)
train_dset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_dloader = DataLoader(train_dset, batch_size=512)

# train
mlp = MLP()
trainer = pl.Trainer(max_epochs=1000,
                     stochastic_weight_avg=True)
trainer.fit(mlp, train_dloader)
trainer.save_checkpoint("./saved_models/6d_norm_stability.ckpt")

# assess accuracy
pred = mlp(torch.Tensor(X_test)).detach().numpy()
pred_train = mlp(torch.Tensor(X_train)).detach().numpy()
print(classification_report(pred > .5, y_test))
print(roc_auc_score(y_test, pred))