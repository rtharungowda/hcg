import config
import glob
import os
import pandas as pd
"""create train and test CSVs.
"""

def create_csv(path_dataset,train=True):
    """creates csv files containing path to image and label

    Args:
        path_dataset (str): path to the folder containing the dataset 
        train (bool, optional): True if the dataset is used for training and False otherwise. Defaults to True.
    """

    folders = os.listdir(path_dataset)
    path = []
    labels = []
    num_folders = 0
    for folder in folders:
        #drop ambiguous characters
        if folder in config.DROP_FOLDER:
            continue
        print(folder)
        num_folders+=1
        folder_path = os.path.join(path_dataset,folder)
        files = glob.glob(folder_path+"/*.png")
        for f in files:
            path.append(f)
            labels.append(config.MAPPING[folder])
    data = {
        'path':path,
        'label':labels
    }

    df = pd.DataFrame(data)
    print(num_folders)
    if train:
        df.to_csv(config.TRAIN_CSV)
    else :
        df.to_csv(config.VAL_CSV)

if __name__ == "__main__":
    create_csv(config.TRAIN_IMAGES)
    create_csv(config.VAL_IMAGES,train=False)