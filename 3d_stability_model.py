import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

# model description
class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(3, 256),
          nn.Dropout(.05),
          nn.ReLU(),
          nn.Linear(256, 128),
                  nn.Dropout(.01),
          nn.ReLU(),
            nn.Linear(128, 128),
          nn.ReLU(),
          nn.Linear(128, 1),
          nn.Sigmoid()
        )
        self.ce = nn.BCELoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# data preparation, 
# use 3d_shear_stability.csv for shear model
df = pd.read_csv('./processed_data/3d_normal_stability.csv')
df['stable'] = df['is_stable'] == 1
X_train, X_test, y_train, y_test = train_test_split(
    df[['exx', 'eyy', 'ezz']].values,
    df['stable'].values,
    test_size=0.1)
y_train = y_train[:, None]
y_test = y_test[:, None]

X_tr_tensor, y_tr_tensor = torch.Tensor(X_train), torch.Tensor(y_train)
train_dset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_dloader = DataLoader(train_dset, batch_size=256)

# model
mlp = MLP()
trainer = pl.Trainer(deterministic=True, max_epochs=10000)
trainer.fit(mlp, train_dloader)
trainer.save_checkpoint("./saved_models/3d_norm_stability.ckpt")

# accessing accuracy
pred = mlp(torch.Tensor(X_test)).detach().numpy()
print(classification_report(pred > .5, y_test))
print(roc_auc_score(y_test, pred))