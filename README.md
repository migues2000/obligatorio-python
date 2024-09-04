# Clasificación de Imágenes con SVM

Este proyecto permite entrenar un modelo de clasificación de imágenes utilizando una Máquina de Vectores de Soporte (SVM) y luego utilizar dicho modelo para clasificar nuevas imágenes. A continuación se describen las dos principales partes del proyecto: entrenamiento del modelo y clasificación de imágenes.

## Entrenamiento del Modelo

El modelo de clasificación de imágenes se entrena utilizando un conjunto de datos de imágenes distribuidas en diferentes categorías. El código para entrenar el modelo se encuentra en `utils.py` bajo la función `getModel`.

### Descripción del Código

1. **Carga o Entrenamiento del Modelo:**

   - La función `getModel()` carga un modelo preentrenado si existe, o entrena un nuevo modelo utilizando imágenes clasificadas en varias categorías.

2. **Preparación de los Datos:**

   - Las imágenes se cargan y se redimensionan a un tamaño fijo de 150x150 píxeles con 3 canales de color (RGB).
   - Las imágenes redimensionadas se aplanan en arreglos 1D y se almacenan en una matriz de datos.
   - Las etiquetas correspondientes a cada categoría se almacenan en una matriz de objetivos.

3. **Entrenamiento del Modelo:**

   - Se utiliza una cuadrícula de parámetros para ajustar los hiperparámetros del modelo SVM mediante `GridSearchCV`.
   - El conjunto de datos se divide en datos de entrenamiento y prueba, y el modelo se entrena en el conjunto de entrenamiento.

4. **Guardado del Modelo:**
   - El modelo entrenado se guarda en un archivo para su uso posterior.

### Ejecución del Entrenamiento

Para entrenar el modelo, asegúrate de que las imágenes estén organizadas en carpetas correspondientes a cada categoría dentro de `DATA_DIR`, y luego ejecuta el script que llama a `getModel()`.

```python
from utils import getModel

model = getModel()
```

## Clasificación de Imágenes

Una vez que el modelo ha sido entrenado, puedes utilizarlo para clasificar nuevas imágenes.

### Descripción del Código

1. **Carga del Modelo:**

   - El modelo entrenado se carga utilizando la función `getModel()`.

2. **Procesamiento de la Imagen:**

   - La imagen que se desea clasificar se carga y se redimensiona al mismo tamaño utilizado durante el entrenamiento (150x150 píxeles).
   - La imagen redimensionada se aplana en un arreglo 1D para que pueda ser procesada por el modelo.

3. **Predicción:**
   - El modelo predice la probabilidad de que la imagen pertenezca a cada categoría y muestra los resultados en la consola.

### Ejecución de la Clasificación

Para clasificar una nueva imagen, ejecuta el siguiente código, proporcionando la ruta de la imagen cuando se solicite:

```python
from utils import getModel
from skimage.io import imread
from skimage.transform import resize

# Carga el modelo entrenado
model = getModel()

# Solicita la ruta de la imagen
url = input('Enter the path of the Image:')

# Lee y redimensiona la imagen
img = imread(url)
img_resized = resize(img, (150, 150, 3))
l = [img_resized.flatten()]

# Predice la categoría de la imagen
probability = model.predict_proba(l)

# Muestra las predicciones
print("The prediction is:")
for ind, val in enumerate(CATEGORIES):
    print(f'{val} = {round(probability[0][ind] * 100, 2)}%')
```

## Requisitos

- Python 3.x
- Bibliotecas Python: `pandas`, `numpy`, `matplotlib`, `joblib`, `scikit-image`, `scikit-learn`

## Estructura del Proyecto

El proyecto sigue la siguiente estructura de carpetas:

```
├── constants.py        # Define las constantes como CATEGORIES y rutas de directorios
├── utils.py            # Contiene la función getModel para entrenar/cargar el modelo
├── main.py             # Contiene un ejemplo de la utilización del modelo
├── data/               # Carpeta donde se almacenan las imágenes de entrenamiento
└── models/             # Carpeta donde se guarda el modelo entrenado
```
