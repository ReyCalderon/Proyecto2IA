from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import pickle
import time
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

historial_texto = []  # Lista para guardar frases detectadas

# -------------------- CARGAR MODELO Y ETIQUETAS --------------------
with open("modelo/clasificador_letras.pkl", "rb") as f:
    data = pickle.load(f)

modelo = tf.keras.models.model_from_json(data['model'])
modelo.set_weights(data['weights'])

with open("modelo/etiquetas_letras.pkl", "rb") as f:
    label_binarizer = pickle.load(f)

# -------------------- FLASK APP --------------------
app = Flask(__name__)

# -------------------- GENERADOR DE CÁMARA --------------------
img_size = 64
ultima_letra = ""
contador_repeticiones = 0
repeticiones_necesarias = 15
texto_completo = ""
ultimo_registro = time.time()

def generar_video():
    global ultima_letra, contador_repeticiones, texto_completo, ultimo_registro
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Detectar mano con MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        if resultado.multi_hand_landmarks:
            for mano in resultado.multi_hand_landmarks:
                # Obtener coordenadas de la mano
                h, w, _ = frame.shape
                puntos = [(int(l.x * w), int(l.y * h)) for l in mano.landmark]

                # Calcular el área que rodea la mano
                x_coords = [p[0] for p in puntos]
                y_coords = [p[1] for p in puntos]
                x1, y1 = max(min(x_coords) - 20, 0), max(min(y_coords) - 20, 0)
                x2, y2 = min(max(x_coords) + 20, w), min(max(y_coords) + 20, h)

                mano_recortada = frame[y1:y2, x1:x2]
                mano_redimensionada = cv2.resize(mano_recortada, (img_size, img_size))
                img = np.expand_dims(mano_redimensionada, axis=0)
                img = img.astype("float32") / 255.0

                # Solo predecir si se detectó y procesó bien la mano
                pred = modelo.predict(img)
                letra_actual = label_binarizer.inverse_transform(pred)[0]

                # Estabilización por repetición
                if letra_actual == ultima_letra:
                    contador_repeticiones += 1
                else:
                    contador_repeticiones = 0
                    ultima_letra = letra_actual

                if contador_repeticiones >= repeticiones_necesarias:
                    tiempo_actual = time.time()
                    if tiempo_actual - ultimo_registro > 2.0:
                        texto_completo += letra_actual
                        ultimo_registro = tiempo_actual
                    contador_repeticiones = 0

                # Mostrar letra en la imagen
                cv2.putText(frame, f"Letra: {letra_actual}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Texto: {texto_completo}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Codificar frame para enviar a navegador
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()

# -------------------- RUTAS FLASK --------------------
@app.route('/')
def index():
    return render_template('index.html', texto=texto_completo)

@app.route('/video')
def video():
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/texto')
def texto():
    global texto_completo, historial_texto
    if texto_completo.strip():
        historial_texto.append(texto_completo.strip())  # guardar
        texto_completo = ""  # reiniciar palabra
    return "<br>".join(historial_texto)  # mostrar todo el historial

@app.route('/reiniciar_palabra')
def reiniciar_palabra():
    global texto_completo
    texto_completo = ""
    return "Palabra reiniciada"

@app.route('/eliminar_ultima')
def eliminar_ultima():
    global texto_completo
    if texto_completo:
        texto_completo = texto_completo[:-1]
    return "Letra eliminada"

# -------------------- INICIO --------------------
if __name__ == '__main__':
    app.run(debug=True)
