import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from skimage.transform import resize
from skimage.io import imread

from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

CATEGORIES = ["DOGS", "CATS"]

DATA_DIR = "data"

MODEL_PATH = 'models/trained_model.pkl'

if os.path.exists(MODEL_PATH):
  print("A trained model already exists, loading...")

  model = joblib.load(MODEL_PATH)
else:
  print("An existing trained model can not be found, a new one will be trained")

  flat_data_array = [] # Input array
  target_array = [] # Output array

  for category in CATEGORIES:
    print(f"Loading {category.lower()} data...")

    path = os.path.join(DATA_DIR, category)

    for img in os.listdir(path):
      img_array = imread(os.path.join(path, img))
      img_resized = resize(img_array, (150, 150, 3))
      flat_data_array.append(img_resized.flatten())
      target_array.append(CATEGORIES.index(category))
    
    print(f"Successfully loaded {category.lower()} data")

  flat_data = np.array(flat_data_array)
  target = np.array(target_array)

  data_frame = pd.DataFrame(flat_data)
  data_frame["Target"] = target

  x = data_frame.iloc[:,:-1] # Input data
  y = data_frame.iloc[:,-1] # Output data

  param_grid = {
      "C": [0.1, 1, 10, 100],
      "gamma": [0.0001, 0.001, 0.01, 0.1, 1],
      "kernel": ["rbf", "poly"]
  }

  svc = svm.SVC(probability = True)
  model = GridSearchCV(svc, param_grid)

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77, stratify=y)
  
  print("Training model...")
  model.fit(x_train, y_train)
  print("Successfully trained model")
  
  joblib.dump(model, MODEL_PATH)

  print("A new model has been trained and saved")

url = input('Enter the path of the Image:')

img = imread(url)

img_resized = resize(img, (150, 150, 3))
l = [img_resized.flatten()]

probability = model.predict_proba(l)

print("The prediction is:")
for ind, val in enumerate(CATEGORIES):
  print(f'{val} = {round(probability[0][ind] * 100, 2)}%')