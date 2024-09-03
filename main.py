import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from skimage.transform import resize
from skimage.io import imread

from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import CATEGORIES
from utils import getModel

model = getModel()

url = input('Enter the path of the Image:')

img = imread(url)

img_resized = resize(img, (150, 150, 3))
l = [img_resized.flatten()]

probability = model.predict_proba(l)

print("The prediction is:")
for ind, val in enumerate(CATEGORIES):
  print(f'{val} = {round(probability[0][ind] * 100, 2)}%')