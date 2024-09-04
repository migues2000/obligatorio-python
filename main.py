import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.io import imread

from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import CATEGORIES  # Importa la lista de categorías para la clasificación
from utils import getModel  # Importa la función getModel que carga o entrena el modelo

# Carga el modelo utilizando la función getModel
model = getModel()

# Solicita al usuario que ingrese la ruta de la imagen a clasificar
url = input('Enter the path of the Image:')

# Lee la imagen desde la ruta proporcionada por el usuario
img = imread(url)

# Redimensiona la imagen a un tamaño fijo de 150x150 píxeles con 3 canales de color (RGB)
img_resized = resize(img, (150, 150, 3))

# Aplana la imagen redimensionada en un array 1D y la coloca en una lista
l = [img_resized.flatten()]

# Predice la probabilidad de que la imagen pertenezca a cada categoría
probability = model.predict_proba(l)

# Muestra las predicciones
print("The prediction is:")
for ind, val in enumerate(CATEGORIES):
  # Imprime el nombre de la categoría y la probabilidad correspondiente, redondeada a dos decimales
  print(f'{val} = {round(probability[0][ind] * 100, 2)}%')