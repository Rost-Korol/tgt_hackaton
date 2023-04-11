from timeit import default_timer as timer
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

import model_builder
from data_setup import SandData, SandDataTest
import engine
import utils

from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
X, y = df.drop(columns='label'), df.label
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42, shuffle=True)

# Set up datasets and dataloaders
train_data = SandData(X, y)
test_data = SandData(X, y)
train_dataloader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=128, shuffle=True)

# initialize model
model = model_builder.Transfer_Cnn(sample_rate=117.2*1000,
                           window_size=512,
                           hop_size=80,
                           mel_bins=64,
                           fmin=0,
                           fmax=58600,
                           classes_num=1,
                           freeze_base=False).to(device)

# Setup loss function and optimizer
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)


# train model
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=300,
             device=device)

# save model state dict
utils.save_model(model, 'test_save.pth')

