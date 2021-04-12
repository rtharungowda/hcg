from utils import load_ckp

import torch
import torch.optim as optim
import numpy as np

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

from model import akbhd, vatch, drklrd, mdl
import config
from dataloader import loader

import time
from tqdm import tqdm

def check_acc():
    # model_ft = akbhd()
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # checkpoint_path_akbhd = [
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd_albu_relu_padded.pt",
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd_albu_relu.pt",
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd_albu.pt",
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd_albu2.pt",
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd.pt",
    #                 ]
    # for p in checkpoint_path_akbhd:
    #     _, _, epoch, val_acc = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
    #     print(f"{val_acc} {p}")
    
    
    # model_ft = drklrd()
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # checkpoint_path_drklrd = [
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/drklrd_albu.pt",
    # ]
    # for p in checkpoint_path_drklrd:
    #     _, _, epoch, val_acc = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
    #     print(f"{val_acc} {p}")
    
    # model_ft = mdl("res18")
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # checkpoint_path_res18 = [
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18_albu_2.pt",
    #                 "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18_albu.pt",
    # ]
    # for p in checkpoint_path_res18:
    #     _, _, epoch, val_acc = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
    #     print(f"{val_acc} {p}")

    model_ft = mdl("res34")
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_res34 = [
                    # "/content/drive/MyDrive/competitions/mosaic-r1/weights/res34_albu_2.pt",
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/res34_albu_26.pt",
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

def ensemble(mdls,dataloaders,dataset_sizes):
    since = time.time()
    acc_p = {'train':[],'val':[]}

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        print(phase)
        running_corrects = 0

        tk = tqdm(dataloaders[phase], total=len(dataloaders[phase]))
        for inputs, labels in tk:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to('cpu')

            all_preds = []
            with torch.no_grad():
                for model in mdls:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    preds = preds.to('cpu')
                    all_preds.append(preds)
            n_preds = torch.cat(all_preds,dim=-1)
            preds,_ = torch.mode(n_preds)

            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        acc_p[phase].append(epoch_acc)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        


if __name__ == "__main__":
    check_acc()

    exit()

    dataloaders,dataset_sizes = loader(use_pretrained=True)

    mdls = []

    model_ft = mdl("res18")
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_res18 = [
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18_albu_2.pt",
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18_albu.pt",
    ]
    for p in checkpoint_path_res18:
        model, _, _, _ = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
        model = model.to(config.DEVICE)
        model.eval()
        mdls.append(model)

    model_ft = mdl("res34")
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_res34 = [
                    "/content/drive/MyDrive/competitions/mosaic-r1/weights/res34_albu_2.pt",
                    # "/content/drive/MyDrive/competitions/mosaic-r1/weights/res34_albu.pt",
    ]
    for p in checkpoint_path_res34:
        model, _, _, _ = load_ckp(p, model_ft, optimizer_ft, config.DEVICE)
        model = model.to(config.DEVICE)
        model.eval()
        mdls.append(model)
        
    ensemble(mdls,dataloaders,dataset_sizes)