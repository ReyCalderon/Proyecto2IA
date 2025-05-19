import cv2
import numpy as np
import tensorflow as tf
import pickle

# -------------------- CONFIG --------------------

img_size = 100
repeticiones_necesarias = 15  # Cuántas veces debe repetirse una letra para aceptarla

# -------------------- CARGA DEL MODELO Y LABELS --------------------

with open("modelo/clasificador_letras.pkl", "rb") as f:
    data = pickle.load(f)

modelo = tf.keras.models.model_from_json(data['model'])
modelo.set_weights(data['weights'])

with open("modelo/etiquetas_letras.pkl", "rb") as f:
    label_binarizer = pickle.load(f)

# -------------------- VARIABLES DE CONTROL --------------------

ultima_letra = ""
contador_repeticiones = 0
palabra = ""

# -------------------- INICIAR CÁMARA --------------------

cam = cv2.VideoCapture(0)
print("[INFO] Mostrá una letra y sostenela para confirmarla. Presioná 'q' para salir.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Preprocesar imagen
    img = cv2.resize(frame, (img_size, img_size))
    img = np.expand_dims(img, axis=0)
    img = img.astype("float32") / 255.0

    # Predicción
    pred = modelo.predict(img)
    letra_actual = label_binarizer.inverse_transform(pred)[0]

    # Verificar repeticiones
    if letra_actual == ultima_letra:
        contador_repeticiones += 1
    else:
        contador_repeticiones = 0
        ultima_letra = letra_actual

    # Confirmar si se repite suficiente
    if contador_repeticiones == repeticiones_necesarias:
        palabra += letra_actual
        contador_repeticiones = 0  # Reiniciar para la siguiente letra
        print(f"[CONFIRMADO] Letra agregada: {letra_actual}")

    # Mostrar en pantalla
    #cv2.putText(frame, f"Letra: {letra_actual}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #cv2.putText(frame, f"Palabra: {palabra}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Clasificador de Letras", frame)

    # Salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
