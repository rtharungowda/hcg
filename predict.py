import cv2

from utils import load_ckp

import torch
import torch.optim as optim
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from custom_model import akbhd
import config
import custom_model

def preprocess(path):
    img = cv2.imread(path,0)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            if img[i][j]>130:
                img[i][j] = 0
            else :
                img[i][j] = 255
    img=np.array(img)/255.
    transform = A.Compose([
                            A.Resize(width=32, height=32),
                            ToTensorV2(),
                        ])
    img = transform(image=img)['image'].float().unsqueeze(0)
    return img

def predict(model, path):
    model.to(config.DEVICE)
    model.eval()
    img = preprocess(path)
    img = img.to(config.DEVICE)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs,1)
    return preds


if __name__ == "__main__":
    path = "/content/drive/MyDrive/competitions/mosaic-r1/test_imgs/ja.jpg"

    model_ft = akbhd()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path = "/content/drive/MyDrive/competitions/mosaic-r1/weights/akbhd.pt"
    model, _, epoch, val_acc = load_ckp(checkpoint_path, model_ft, optimizer_ft, config.DEVICE)
    preds = predict(model, path)
    print(preds)