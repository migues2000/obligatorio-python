import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from skimage.transform import resize
from skimage.io import imread

from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import CATEGORIES, DATA_DIR, MODELS_PATH

def getModel():
  # Verifica si ya existe un modelo entrenado en la ruta especificada
  if os.path.exists(MODELS_PATH):
    print("Ya existe un modelo entrenado, cargando...")

    # Cargar el modelo utilizando joblib
    model = joblib.load(MODELS_PATH)

    print("Modelo cargado exitosamente")

    # Retorna el modelo cargado
    return model
  else:
    print("No se ha encontrado un modelo entrenado existente, se entrenará uno nuevo")

    flat_data_array = []  # Inicializa una lista vacía para almacenar los datos de las imágenes aplanadas
    target_array = []  # Inicializa una lista vacía para almacenar las etiquetas de los objetivos

    # Itera sobre cada categoría en la lista de categorías
    for category in CATEGORIES:
      print(f"Cargando datos de {category.lower()}...")

      # Define la ruta a las imágenes para la categoría actual
      path = os.path.join(DATA_DIR, category)

      # Itera sobre cada archivo de imagen en el directorio de la categoría
      for img in os.listdir(path):
        # Lee la imagen como un arreglo
        img_array = imread(os.path.join(path, img))
        # Redimensiona la imagen a un tamaño fijo (150x150 píxeles, 3 canales)
        img_resized = resize(img_array, (150, 150, 3))
        # Aplana la imagen redimensionada y la agrega a la lista de datos de entrada
        flat_data_array.append(img_resized.flatten())
        # Agrega el índice de la categoría al array de objetivos
        target_array.append(CATEGORIES.index(category))

      print(f"Datos de {category.lower()} cargados con éxito")

    # Convierte la lista de imágenes aplanadas y las etiquetas de los objetivos en arreglos numpy
    flat_data = np.array(flat_data_array)
    target = np.array(target_array)

    # Crea un DataFrame con los datos de las imágenes aplanadas
    data_frame = pd.DataFrame(flat_data)
    # Agrega una columna al DataFrame para las etiquetas de los objetivos
    data_frame["Target"] = target

    # Separa el DataFrame en características de entrada (x) y etiquetas de salida (y)
    x = data_frame.iloc[:, :-1]  # Todas las columnas excepto la última
    y = data_frame.iloc[:, -1]  # La última columna (Target)

    # Define una cuadrícula de parámetros para que GridSearchCV optimice los hiperparámetros del SVM
    param_grid = {
      "C": [0.1, 1, 10, 100],  # Parámetro de regularización
      "gamma": [0.0001, 0.001, 0.01, 0.1, 1],  # Coeficiente del kernel
      "kernel": ["rbf", "poly"]  # Tipo de kernel (Radial Basis Function, Polinómico)
    }

    # Crea un modelo SVM con estimaciones de probabilidad habilitadas
    svc = svm.SVC(probability = True)
    # Usa GridSearchCV para encontrar la mejor combinación de hiperparámetros
    model = GridSearchCV(svc, param_grid)

    # Divide los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77, stratify = y)

  # Entrena el modelo con los datos de entrenamiento
  print("Entrenando modelo...")
  model.fit(x_train, y_train)
  print("Modelo entrenado con éxito")

  # Guarda el modelo entrenado en la ruta especificada usando joblib
  joblib.dump(model, MODELS_PATH)

  print("Se ha entrenado y guardado un nuevo modelo")

  # Retorna el modelo entrenado
  return model
