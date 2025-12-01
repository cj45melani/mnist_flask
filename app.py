from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
import re
import tensorflow as tf

app = Flask(__name__)

# Cargar modelo entrenado
modelo = tf.keras.models.load_model("model.h5")

# Página principal
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint para predicción del canvas
@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    data = request.get_json()

    if "image" not in data:
        return jsonify({"error": "No se envió la imagen"}), 400

    # Convertir base64 a imagen
    image_data = re.sub('^data:image/.+;base64,', '', data["image"])
    img_bytes = base64.b64decode(image_data)

    # Convertir a numpy array
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # Preprocesamiento tipo MNIST
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predicción
    pred = modelo.predict(img)
    num = int(np.argmax(pred))
    confianza = float(np.max(pred))

    return jsonify({"prediccion": num, "confianza": confianza})


if __name__ == "__main__":
    app.run(debug=True)
