import cv2
import glob
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms, models

from segmentation import perform_segmentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ckp(checkpoint_fpath, model, optimizer, device):

    checkpoint = torch.load(checkpoint_fpath,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_acc = checkpoint['valid_acc'] 
    return model, optimizer, checkpoint['epoch'], valid_acc

def preprocess(img):
    img = np.array(img)
    h = img.shape[0]
    w = img.shape[1]
    new_img = np.zeros((h,w,3))
    for i in range(h):
        for j in range(w):
            if img[i][j]>130:
                new_img[i][j][0]=img[i][j]
                new_img[i][j][1]=img[i][j]
                new_img[i][j][2]=img[i][j]
            else :
                new_img[i][j][0]=img[i][j]
                new_img[i][j][1]=img[i][j]
                new_img[i][j][2]=img[i][j]
    transform = A.Compose([
                    A.Resize(width=224, height=224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
    img = transform(image=new_img)['image'].float().unsqueeze(0)
    return img

def predict_charac(img):
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 27)

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    checkpoint_path_res34 = "/content/drive/MyDrive/competitions/mosaic-r1/weights/res34_albu_26.pt"
    model, _, _, _ = load_ckp(checkpoint_path_res34, model_ft, optimizer_ft, DEVICE)
    model = model.to(DEVICE)
    model.eval()

    img = preprocess(img)
    img = img.to(DEVICE)
    all_preds = []
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    return preds


if __name__ == "__main__":
    path = "/content/drive/MyDrive/Mosaic1 sample/samay2.jpg"
    print(path)
    imgs = perform_segmentation(path)
    for img in imgs:
        preds = predict_charac(img)
        print(preds)