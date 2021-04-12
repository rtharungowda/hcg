import os
import torch
"""contains all hyperparameters and paths required.
"""

LR = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 27
TRAIN_IMAGES = "/content/drive/MyDrive/competitions/mosaic-r1/dataset/Train"
VAL_IMAGES = "/content/drive/MyDrive/competitions/mosaic-r1/dataset/Test"
TRAIN_CSV = '/content/drive/MyDrive/competitions/mosaic-r1/train_26.csv'
VAL_CSV = '/content/drive/MyDrive/competitions/mosaic-r1/val_26.csv'

#2,3,19,5,15,14,30,17,24.36

MAPPING = {
    'character_10_yna':7,
    'character_11_taamatar':8,
    'character_12_thaa':9, 
    'character_13_daa':10,  
    'character_16_tabala':11, 
    'character_18_da':12, 
    'character_1_ka':1, 
    'character_20_na':13, 
    'character_21_pa':14, 
    'character_22_pha':15, 
    'character_23_ba':16,  
    'character_25_ma':17, 
    'character_26_yaw':18, 
    'character_27_ra':19, 
    'character_28_la':20, 
    'character_29_waw':21,   
    'character_31_petchiryakha':22, 
    'character_32_patalosaw':23, 
    'character_33_ha':24, 
    'character_34_chhya':25, 
    'character_35_tra':26, 
    'character_4_gha':2,  
    'character_6_cha':3, 
    'character_7_chha':4, 
    'character_8_ja':5, 
    'character_9_jha':6,
}

DROP_FOLDER = ['character_2_kha','character_3_ga','character_19_dha','character_5_kna','character_15_adna','character_14_dhaa','character_30_motosaw',
                'character_17_tha','character_24_bha','character_36_gya']

if __name__ == "__main__":
    print(NUM_CLASSES)