from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from main import classes


app = Flask(__name__)

# Załadowanie modelu
model = load_model("braille_model.h5")
save_path = "savepath/zdjecieTestowane"

# Przetworzenie obrazu testowanego
def preprocess_image(image):
    processed_image = cv2.resize(image, (28, 28))
    save_path_with_extension = save_path + ".jpg"
    cv2.imwrite(save_path_with_extension, processed_image)
    return processed_image

# Predykcja klasy obrazu
def predict_class(image):
    preprocessed_image = preprocess_image(image)
    input_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    return predicted_class

# Route dla strony domowej
@app.route('/')
def home():
    return render_template('index.html')

# Route dla uploadu obrazu i wykonania predykcji
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    predicted_class = predict_class(image)
    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
