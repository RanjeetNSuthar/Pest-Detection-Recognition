## 1. Problem

Given an image of any crop field, the model should be able to detect the presence of any pest inside the crops and at the same time recognize its type so that specific action can be taken to control its further spread.

## 2. Data

The data we're using is from Kaggle, shared by Ratul Mahjabin.
This dataset was proposed and accepted by the authors={Xiaoping Wu ,Chi Zhan, Yukun Lai, Ming-Ming Cheng and Jufeng Yang} in CVPR 2019

https://www.kaggle.com/rtlmhjbn/ip02-dataset

Dataset has **75,222 images** and average size of 737 samples per class. The dataset has a split of 6:1:3(train:val:test).

The Overall Data is divided into 3 Directories namely train,test and val.
Each Directory contains 102 subdirectories each containing multiple images of one of the respective class of pests. hence all the 102 subdirectories(named from 0 to 101) together contains images of each of the type of all the 102 classes of pests.
**labels** for each class are provided in **classes.txt** file

**Note:**
We have added one more subdirectory named 103 in each of the 3 directories each containing images of normal crops free of any kind of pest inorder to detect healthy crops.This makes the class count = 103 (0 to 102)


## 3. Evaluation

The model detect and recognize the presence of pest with an accuracy >= 80%

## 4. Features

Some information about the data:
* We're dealing with images (unstructured data) so it's probably best we use deep learning/transfer learning.
* There are 103 classes (this means there are 102 different pest species + 1 class representing healthy crops free of any kind of pest).
* There are around 45,000+ images in the training set (corresponding labels in train.txt)
* There are around 7,000+ images in the val set (corresponding labels in val.txt).
* There are around 22,000+ images in the test set (corresponding labels in test.txt).
## 5. Execution

To run the model run the recognizer.py file after configuring it according to your environment which will use the trained model and provide the results

