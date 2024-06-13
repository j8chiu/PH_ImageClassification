# TopoClassification

PyTorch implementation of the DSC214 Final Project (the original PHG-Net and ViT with all-attention part)

## Environment

The required environment settings are the same as main branch

## 1. Data

download [ISIC](https://challenge.isic-archive.com/) dataset and put them into the directory home\data\ISIC, remember to split training and testing set

## 2. Persistent Diagram

run `pd_pl.py` file to get the persistent homology information for training/test datasets. You might need to change certain settings

## 3.Training

run `phgnet_train.py` to reproduce the results from original PHG-Net paper, this py file contains the training process for a ResNet50 + Topo feature model

run `topovit.py` for realizing the all-attention mechanism introduced in our report, it contains a pre-trained ViT model that you might need to download beforehand
