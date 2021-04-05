import config 
import dataloader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from tqdm import tqdm 

# from efficientnet_pytorch import EfficientNet
# import pretrainedmodels
from dataloader import loader
from utils import save_ckp, plot

# scaler = torch.cuda.amp.GradScaler()

def train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, checkpoint_path, num_epochs=25):
    print(f"saving to {checkpoint_path}")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    loss_p = {'train':[],'val':[]}
    acc_p = {'train':[],'val':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            tk = tqdm(dataloaders[phase], total=len(dataloaders[phase]))
            for inputs, labels in tk:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scaler.scale(loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # print("running loss ",running_loss)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_p[phase].append(epoch_loss)
            acc_p[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'valid_acc': best_acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                # checkpoint_path = "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18.pt"
                print(f"saving to {checkpoint_path}")
                save_ckp(checkpoint, checkpoint_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    plot(loss_p,acc_p,num_epochs)

    return model, best_acc

def mdl(type):
    if type == "res18":
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)

        return model_ft
    
    elif type == "res50":
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

        return model_ft

    elif type == "eff-b6":
        model = EfficientNet.from_name('efficientnet-b6', num_classes=2)
        return model

    elif type == "eff-b3":
        model = EfficientNet.from_name('efficientnet-b3', num_classes=2)
        return model
    
    elif type == "dns201":
        model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet201', pretrained=False)
        return model

    elif type == "rsnxt-50":
        model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](num_classes=2,pretrained=None)
        return model

if __name__ == '__main__':
    dataloaders,dataset_sizes = loader()

    model_ft = mdl("res18")
    model_ft = model_ft.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    checkpoint_path = "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18.pt"

    model_ft, best_acc = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes, checkpoint_path, num_epochs=config.NUM_EPOCHS)