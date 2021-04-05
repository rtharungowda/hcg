import config 
import train

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import copy
from dataloader import loader
from utils import save_ckp, plot
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

class character(Dataset):
    def __init__ (self, dataframe, transform):
        self.df = dataframe
        self.tf = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,ind):
        x = Image.open(self.df['path'].iloc[ind])
        x = np.array(x)/255.
        x = self.tf(image=x)['image']
        x = x.float()

        y = self.df['label'].iloc[ind]
        return x, y
    
def albu():
    transform ={"train":A.Compose([
                                    A.Resize(width=32, height=32),
                                    ToTensorV2(),
                                ]),
                "val":A.Compose([
                                    A.Resize(width=32, height=32),
                                    ToTensorV2()
                                ])
                } 
    return transform

def loader():
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    dfs = {
        'train':train_df,
        'val' :val_df
    }

    transform = albu()

    img_datasets = {
        x:character(dfs[x],transform[x])
        for x in ['train','val']
    }

    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}

    print(dataset_sizes)

    dataloaders = {
        x: DataLoader(img_datasets[x], batch_size=config.BATCH_SIZE,shuffle=True, num_workers=2)
        for x in ['train', 'val']}

    return dataloaders,dataset_sizes

class akbhd(nn.Module):
    def __init__(self):
        super(akbhd, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=(5,5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=(5,5))
        self.maxpool2 = nn.MaxPool2d(kernel_size=5,stride=5)
        self.linear1 = nn.Linear(64*2*2,config.NUM_CLASSES)
    
    def forward(self,x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        # x = F.ReLU(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        # x = F.ReLU(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        return x


if __name__ == '__main__':
    dataloaders,dataset_sizes = loader()

    model_ft = akbhd()
    model_ft = model_ft.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    checkpoint_path = "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd.pt"
    model_ft, best_acc = train.train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes, checkpoint_path, num_epochs=config.NUM_EPOCHS)