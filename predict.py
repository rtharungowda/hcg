import cv2
from PIL import Image, ImageOps

from utils import load_ckp

import torch
import torch.optim as optim
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import akbhd, mdl
import config

def preprocess(path,pretrained):
    if pretrained == True:
        img = Image.open(path)
        img = ImageOps.grayscale(img)
        img = np.array(img)
        h = img.shape[0]
        w = img.shape[1]
        new_img = np.zeros((h,w,3))
        for i in range(h):
            for j in range(w):
                if img[i][j]>130:
                    img[i][j] = 0
                    new_img[i][j][0]=img[i][j]
                    new_img[i][j][1]=img[i][j]
                    new_img[i][j][2]=img[i][j]
                else :
                    img[i][j] = 255
                    new_img[i][j][0]=img[i][j]
                    new_img[i][j][1]=img[i][j]
                    new_img[i][j][2]=img[i][j]
        img = new_img
        cv2.imwrite("/content/drive/MyDrive/competitions/mosaic-r1/test_imgs/ma_res.jpg",img)
        transform = A.Compose([
                    # A.InvertImg(always_apply=False, p=0.5),
                    A.Resize(width=224, height=224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
    else :
        img = cv2.imread(path,0)
        h = img.shape[0]
        w = img.shape[1]
        for i in range(h):
            for j in range(w):
                if img[i][j]>130:
                    img[i][j] = 0
                else :
                    img[i][j] = 255
        img=np.array(img)
        img/=255.
        transform = A.Compose([
                                A.Resize(width=32, height=32),
                                ToTensorV2(),
                            ])

    img = transform(image=img)['image'].float().unsqueeze(0)
    return img

def predict(model, path, pretrained):
    model.to(config.DEVICE)
    model.eval()
    img = preprocess(path, pretrained)
    img = img.to(config.DEVICE)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs,1)
    return preds


if __name__ == "__main__":
    path = "/content/drive/MyDrive/competitions/mosaic-r1/test_imgs/ma.jpg"

    # model_ft = akbhd()
    model_ft = mdl("res18")
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path = "/content/drive/MyDrive/competitions/mosaic-r1/weights/res18_albu.pt"
    model, _, epoch, val_acc = load_ckp(checkpoint_path, model_ft, optimizer_ft, config.DEVICE)
    print(val_acc)
    preds = predict(model, path, pretrained=True)
    print(preds)