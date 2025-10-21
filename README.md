# COVID 19 Radiography

## Overview

This project implements a deep learning pipeline for the classification of medical chest X-ray images into four categories:
- COVID
- Lung_Opacity
- Normal
- Viral_Pneumonia

14816 images are used for training, 3172 images for validation and 3179 images for testing.
The goal is to create an accurate model capable of distinguishing between these conditions. For the implementation Tensorflow is used.
The data is available on Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

## Project Steps

**Exploratory Data Analysis (EDA)**

The dataset is analyzed to gain an overview of class distributions, and potential data imbalances.
Visualizations help understand the dataset and inform preprocessing strategies.

**Model Setup**

A pretrained ResNet50 model from tf.keras.applications is used with ImageNet weights. In total the model has 24,122,500 parameters.
On top of the base model global average pooling, batch normalization, dropout, and dense layers are added.

**Training Strategy**

- Step 1: Feature Extraction

    All layers of the model are frozen to leverage pretrained features. Only the top layers are trained on the dataset.

- Step 2: Fine-tuning

    The upper 100 layers are unfrozen to improve performance. The learning rate is reduced by factor 10 for stable fine-tuning.

**Evaluation & Visualization**

Classification metrics such as accuracy, precision, recall, cosine-simarlity, F1-score and AUC are computed.
The results are visualized through a confusion matrix, an accuracy-loss plot and a ROC-curve.

|  Class              | Precision | Recall | F1-Score | Support |
|:--------------------|:---------:|:------:|:--------:|--------:|
| **COVID**           |   0.97    |  0.96  |   0.96   |  543    |
| **Lung Opacity**    |   0.94    |  0.93  |   0.93   |  903    |
| **Normal**          |   0.95    |  0.94  |   0.95   |  1530   |
| **Viral Pneumonia** |   0.87    |  1.00  |   0.93   |  203    |
| **Accuracy**        |           |        |   0.95   |  3179   |
| **Macro avg**       |   0.93    |  0.96  |   0.94   |  3179   |
| **Weighted avg**    |   0.95    |  0.95  |   0.95   |  3179   |
