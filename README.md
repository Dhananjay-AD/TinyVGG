# TinyVGG

 This repository contains a neural network model inspired by TinyVGG. It include full data preparation containing three classes ['chicken_curry','chocolate_cake','ice_cream'] , model building, training and evaluation forming a complete machine learning pipeline

 ### Image

Following image represent the structure of neural network model. The only difference is, there are there classes or there output terms in the model built in this repository

![TinyVGG](./images/a.png)

## Installation instruction for users

- Clone the repo
    git clone https://github.com/Dhananjay-AD/TinyVGG.git
- check for requirements
    `pip install -r requirements.txt`
- place the custom dataset in the following way
    - TinyVGG/dataset/custom dataset/
    - training dataset/
        - class1/
        - class2/
        - class3/
    - testing dataset/
        - class1/
        - class2/
        - class3/

## Training the neural network
Run `train.py` with arguments 
- `--name` for naming the model
- `--use_pretrained_model` for using pretrained model for better results
- `--lr` for selecting learning rate
- `--epochs` for selecting number of epochs to train
- `--batch_size` for selecting batch size
- `--loss_curve` to visualize loss curve


