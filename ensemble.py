from utils import load_ckp

import torch
import torch.optim as optim
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import akbhd, vatch, drklrd, mdl
import config

def check_acc():
    model_ft = akbhd()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_akbhd = [
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd_albu_relu_padded.pt",
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd_albu_relu.pt",
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd_albu.pt",
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd_albu2.pt",
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd.pt",
                    ]
    for p in checkpoint_path_akbhd:
        _, _, epoch, val_acc = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
        print(f"{val_acc} {p}")
    
    
    model_ft = drklrd()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_drklrd = [
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/drklrd_albu.pt",
    ]
    for p in checkpoint_path_drklrd:
        _, _, epoch, val_acc = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
        print(f"{val_acc} {p}")
    
    model_ft = mdl("res18")
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_res18 = [
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18_albu_2.pt",
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18_albu.pt",
    ]
    for p in checkpoint_path_res18:
        _, _, epoch, val_acc = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
        print(f"{val_acc} {p}")

    model_ft = mdl("res34")
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_res34 = [
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/res34_albu_2.pt",
                    # "/content/drive/MyDrive/competitions/mosaic-r1/weights/res34_albu.pt",
    ]
    for p in checkpoint_path_res34:
        _, _, epoch, val_acc = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
        print(f"{val_acc} {p}")

    # model_ft = vatch()
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # checkpoint_path_vatch = [
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/vatch_albu.pt",
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/vatch_albu2.pt",
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/vatch.pt"
    # ]
    # for p in checkpoint_path_vatch:
    #     _, _, epoch, val_acc = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
    #     print(f"{val_acc} {p}")

if __name__ == "__main__":
    check_acc()