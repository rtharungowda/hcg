This file describes the training procedure involved in optical character recognition of hindi characters
The dataset can be found here https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset
Edit files and execute files in the following order and change paths where ever necessary

utils.py - contains all the necessary plot, load and save model helper functions.

config.py - contains all the necessary hyperparameters and paths to dataset and also the character to numeric mapping.
pre_csv.py - creates a csv file for train and test, which contains the path to images along with its label.
dataloader.py - implements augmentation and preproceesing on custom dataset and clubs images into batches using dataloader.
model.py (used for experimentaiton) - contains pretrained and custom models which were used for experimentaiton. We finally decided to use ResNet34 pretrained on Imagenet dataset, which gave a 99.75 validation accuracy.
train.py - train model

remove.sh - removes numberic character images from dataset extracted
setup.sh - run it if you are using colab to run the files

