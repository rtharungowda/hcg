import pandas as pd
import numpy as np
import os
import glob
import config
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

class character_pre(Dataset):
    def __init__ (self, dataframe, transform):
        """custom dataset for pytorch

        Args:
            dataframe (pandas.DataFrame): pandas dataframe of the csv file for training and testing,
            transform (albumentations.Compose): augmentations and preprocessing to be applied on images.
        """
        self.df = dataframe
        self.tf = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,ind):
        x = Image.open(self.df['path'].iloc[ind])
        x = x.convert('RGB')
        x = np.array(x)
        # cv2.imwrite(f"/content/drive/MyDrive/competitions/mosaic-r1/test_imgs/{ind}_dataloader.png",x)
        x = self.tf(image=x)['image']
        x = x.float()

        y = self.df['label'].iloc[ind]
        return x, y
    
def albu():
    """augmentations and preprocessing on images

    Returns:
        albumentations.Compose: alubmentations transforms
    """
    transform ={"train":A.Compose([
                            A.OneOf([
                                A.ShiftScaleRotate(always_apply=False, p=0.3, shift_limit=(0.0, 0.0), scale_limit=(0.0, 0.0), rotate_limit=(-10, 10), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                                A.ShiftScaleRotate(always_apply=False, p=0.3, shift_limit=(-0.1, 0.1), scale_limit=(0.0, 0.0), rotate_limit=(0, 0), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                                A.ShiftScaleRotate(always_apply=False, p=0.3, shift_limit=(0.0, 0.0), scale_limit=(-0.1, 0.1), rotate_limit=(0, 0), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                                A.ShiftScaleRotate(always_apply=False, p=0.1, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), rotate_limit=(-10, 10), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                            ],p=1.0),
                            # A.InvertImg(always_apply=False, p=0.5),
                            A.Resize(width=224, height=224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                        ]),
        "val":A.Compose([
                            # A.InvertImg(always_apply=False, p=0.5),
                            A.Resize(width=224, height=224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                        ])
        } 
    return transform

def loader():
    """Dataloaders for pytorch training

    Returns:
        dataloaders : dictionary of train and validation dataloaders
        dataset_sizes : dictionary of train and validation dataset sizes
    """
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    dfs = {
        'train':train_df,
        'val' :val_df
    }

    transform = albu()

    img_datasets = {
        x:character_pre(dfs[x],transform[x])
        for x in ['train','val']
    }
   
    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}

    print(dataset_sizes)

    dataloaders = {
        x: DataLoader(img_datasets[x], batch_size=config.BATCH_SIZE,shuffle=True, num_workers=2)
        for x in ['train', 'val']
        }

    return dataloaders,dataset_sizes

if __name__=='__main__':
    train_df = pd.read_csv(config.TRAIN_CSV)
    tf = A.Compose([
                    A.OneOf([
                        A.ShiftScaleRotate(always_apply=False, p=0.3, shift_limit=(0.0, 0.0), scale_limit=(0.0, 0.0), rotate_limit=(-10, 10), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                        A.ShiftScaleRotate(always_apply=False, p=0.3, shift_limit=(-0.1, 0.1), scale_limit=(0.0, 0.0), rotate_limit=(0, 0), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                        A.ShiftScaleRotate(always_apply=False, p=0.3, shift_limit=(0.0, 0.0), scale_limit=(-0.1, 0.1), rotate_limit=(0, 0), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                        A.ShiftScaleRotate(always_apply=False, p=0.1, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), rotate_limit=(-10, 10), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                    ],p=1.0),
                    A.Resize(width=32, height=32),
                    ToTensorV2(),
                ])
    dataset = character_pre(train_df,tf)
    x,y = dataset.__getitem__(5)