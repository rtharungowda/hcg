import config
import glob
import os
import pandas as pd

def create_csv(path_dataset,train=True):
    folders = os.listdir(path_dataset)
    path = []
    labels = []
    num_folders = 0
    for folder in folders:
        if folder in config.DROP_FOLDER:
            continue
        print(folder)
        num_folders+=1
        folder_path = os.path.join(path_dataset,folder)
        files = glob.glob(folder_path+"/*.png")
        for f in files:
            # print(f,folder)
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