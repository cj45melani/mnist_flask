from flask import Flask, render_template, request
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# ----------------------------------------
# ENTRENAR MODELO SI NO EXISTE
# ----------------------------------------
if not os.path.exists("model.h5"):
    print("Entrenando modelo, espera...")
    (ds_train, ds_test), info = tfds.load(
        "mnist", split=["train", "test"], as_supervised=True, with_info=True
    )

    def preprocessing(img, label):
        img = tf.image.resize(img, (128, 128))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    train_ds = ds_train.map(preprocessing).shuffle(10000).batch(32)
    test_ds = ds_test.map(preprocessing).batch(32)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=3, validation_data=test_ds)
    model.save("model.h5")
    print("Modelo guardado como model.h5")

# ----------------------------------------
# CARGAR MODELO YA ENTRENADO
# ----------------------------------------
model = tf.keras.models.load_model("model.h5")

# ----------------------------------------
# Rutas de Flask
# ----------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(img_path)

        # Procesar imagen
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=(128,128))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        prediction = int(np.argmax(pred))

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
