import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model("final_model.h5")

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 128

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # important
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 128, 128, 3)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = preprocess(filepath)
            pred = model.predict(img)

            index = np.argmax(pred)
            prediction = labels[index]

            img_path = filepath

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True, port=5004)