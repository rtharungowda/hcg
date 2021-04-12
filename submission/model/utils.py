import pandas as pd
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch

def plot(loss_p,acc_p,epochs):
    """plot loss, accuracy of train and validation

    Args:
        loss_p (array): array of loss values
        acc_p (array): array of accuracy values
        epochs (array): array of epoch values
    """
    x = [i for i in range(epochs)]
    plt.plot(x,loss_p['train'],color='red', marker='o')
    plt.title('Train loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True) 
    plt.savefig('/content/train_loss.png')
    plt.clf()

    plt.plot(x, loss_p['val'],color='red', marker='o')
    plt.title('Val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True) 
    plt.savefig('/content/val_loss.png')
    plt.clf()
    
    #acc
    plt.plot(x, acc_p['train'],color='red', marker='o')
    plt.title('Train acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid(True) 
    plt.savefig('/content/train_acc.png')
    plt.clf()

    plt.plot(x, acc_p['val'],color='red', marker='o')
    plt.title('Val acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid(True) 
    plt.savefig('/content/val_acc.png')
    plt.clf()

def save_ckp(state, checkpoint_path):
    """save model

    Args:
        state (checkpoint dictionary): dictionary of model, optimizer, epoch and validation accuracy
        checkpoint_path (string): path to save model
    """
    f_path = checkpoint_path
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer, device):
    """load model from path

    Args:
        checkpoint_fpath (str): path of saved model
        model (torch.model): pytorch model
        optimizer (torch.optim): pytorch optimizer
        device (torch.device): load model on device

    Returns:
        torch.model: pytorch model
        torch.optim: pytorch optimizer
        int: epoch number 
        float: validation accuracy
    """
    checkpoint = torch.load(checkpoint_fpath,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_acc = checkpoint['valid_acc'] 
    return model, optimizer, checkpoint['epoch'], valid_acc
