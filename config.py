import os
import torch

LR = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 37
TRAIN_IMAGES = "/content/drive/MyDrive/competitions/mosaic-r1/dataset/Train"
VAL_IMAGES = "/content/drive/MyDrive/competitions/mosaic-r1/dataset/Test"
TRAIN_CSV = '/content/drive/MyDrive/competitions/mosaic-r1/train.csv'
VAL_CSV = '/content/drive/MyDrive/competitions/mosaic-r1/val.csv'

MAPPING = {
    'character_10_yna':10,
    'character_11_taamatar':11,
    'character_12_thaa':12, 
    'character_13_daa':13, 
    'character_14_dhaa':14, 
    'character_15_adna':15, 
    'character_16_tabala':16, 
    'character_17_tha':17,
    'character_18_da':18, 
    'character_19_dha':19, 
    'character_1_ka':1, 
    'character_20_na':20, 
    'character_21_pa':21, 
    'character_22_pha':22, 
    'character_23_ba':23, 
    'character_24_bha':24, 
    'character_25_ma':25, 
    'character_26_yaw':26, 
    'character_27_ra':27, 
    'character_28_la':28, 
    'character_29_waw':29, 
    'character_2_kha':2, 
    'character_30_motosaw':30, 
    'character_31_petchiryakha':31, 
    'character_32_patalosaw':32, 
    'character_33_ha':33, 
    'character_34_chhya':34, 
    'character_35_tra':35, 
    'character_36_gya':36, 
    'character_3_ga':3, 
    'character_4_gha':4, 
    'character_5_kna':5, 
    'character_6_cha':6, 
    'character_7_chha':7, 
    'character_8_ja':8, 
    'character_9_jha':9,
}

if __name__ == "__main__":
    print(NUM_CLASSES)