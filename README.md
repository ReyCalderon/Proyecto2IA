# Traductor de Lenguaje de Señas a Texto

Este proyecto implementa un sistema de reconocimiento de lenguaje de señas manuales (letras) utilizando visión por computadora e inteligencia artificial. El sistema detecta señas mediante la cámara en tiempo real y las traduce a texto a través de una interfaz web interactiva.

## Tecnologías utilizadas

- Python
- Flask
- OpenCV
- TensorFlow 
- Scikit-learn
- HTML, CSS, JavaScript

## Estructura del proyecto

proyecto/
├── modelo/
│ ├── clasificador_letras.pkl
│ └── etiquetas_letras.pkl
├── dataset/ 
├── src/
│ ├── app.py
│ ├── entrenar_modelo.py
│ ├── templates/
│ │ └── index.html
│ └── static/
│ └── style.css
└── README.md

## Instalación y ejecución

1. Cloná el repositorio o copiá los archivos.
2. Creá un entorno virtual e instalá las dependencias:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
3 Ejecutá la aplicación
    python src/app.py
4 Abrir el navegador en este caso http://127.0.0.1:5000
   
## ¿Cómo funciona?
Se capturan imágenes con la cámara web.
Se preprocesan y clasifican usando un modelo CNN entrenado previamente.
Cada letra detectada se acumula como texto, que se muestra al usuario.
La cámara puede ser activada o desactivada desde la interfaz web.
El texto puede ser guardado con un botón y reiniciado automáticamente.

## Reconocimiento actual
El sistema está entrenado para detectar letras del alfabeto en lenguaje de señas. Se implementó un filtro de estabilidad y un sistema de "cooldown" para evitar repeticiones rápidas no deseadas.
