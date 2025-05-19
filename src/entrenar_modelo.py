import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pickle

# -------------------- CONFIGURACIONES --------------------

dataset_path = "dataset"  # Asumiendo que ejecutas desde la raíz del proyecto
img_size = 64             # Tamaño al que se redimensionan las imágenes
X = []
y = []

# Crear carpeta 'modelo' si no existe
os.makedirs("modelo", exist_ok=True)

# -------------------- CARGA DE IMÁGENES --------------------

print("[INFO] Cargando imágenes...")
for etiqueta in os.listdir(dataset_path):
    carpeta = os.path.join(dataset_path, etiqueta)
    if not os.path.isdir(carpeta):
        continue
    for img_name in os.listdir(carpeta):
        img_path = os.path.join(carpeta, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(etiqueta)
        except:
            print(f"[WARN] No se pudo leer la imagen: {img_path}")

print(f"[INFO] Total de imágenes cargadas: {len(X)}")
print(f"[INFO] Total de clases detectadas: {len(set(y))}")

# -------------------- PREPROCESAMIENTO --------------------

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

lb = LabelBinarizer()
y = lb.fit_transform(y)

# Guardar codificador de etiquetas
with open("modelo/etiquetas_letras.pkl", "wb") as f:
    pickle.dump(lb, f)

# Separar entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- MODELO CNN --------------------

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(lb.classes_), activation='softmax')
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------- ENTRENAMIENTO --------------------

print("[INFO] Entrenando modelo...")
modelo.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# -------------------- GUARDADO DEL MODELO --------------------

with open("modelo/clasificador_letras.pkl", "wb") as f:
    pickle.dump({
        'model': modelo.to_json(),
        'weights': modelo.get_weights()
    }, f)

print("✅ Modelo y etiquetas guardados correctamente.")
print("[INFO] Clases:", lb.classes_)