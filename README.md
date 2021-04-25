# Hindi optical character recognition

## Installation and setup

+ If you are cloning locally then install dependencies by running:

```shell
pip3 install -r requirements.txt
```

+ If you are using colab then install additional dependencies by running:

```shell
sh setup.sh
```

## Dataset

+ The dataset can be found here https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset
+ After extracting files run `setup.sh` with variables *TRAIN_PATH* and *TEST_PATH* having their respective paths. This is to remove classes containg digits which we wont train on.

<div align="center">
<img src="vis.png" width="70%"></br>
<span>A batch of training images</span>
</div>

## File Description

Edit files and execute files in the following order and change paths where ever necessary

+ utils.py - contains all the necessary plot, load and save model helper functions.

+ config.py - contains all the necessary hyperparameters and paths to dataset and also the character to numeric mapping.
+ pre_csv.py - creates a csv file for train and test, which contains the path to images along with its label.
+ dataloader.py - implements augmentation and preproceesing on custom dataset and clubs images into batches using dataloader.
+ model.py (used for experimentaiton) - contains pretrained and custom models which were used for experimentaiton. We finally decided to use ResNet34 pretrained on Imagenet dataset, which gave a 99.75 validation accuracy.
+ train.py - train model

+ remove.sh - removes numberic character images from dataset extracted
+ setup.sh - run it if you are using colab to run the files
