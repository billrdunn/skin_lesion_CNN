import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)


def get_model():
    global model
    model = load_model('mobilenet_224x224_layers=60.h5')
    print(" * Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":  # convert to RGB format if not already
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


print(" * Loading Keras model...")
get_model()


@app.route("/predict_with_visuals", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image).tolist() # pass to model for prediction and convert to python list

    response = {  # python dictionary
        'prediction': { # dictionary within dictionary
            'akiec': prediction[0][0],
            'bcc': prediction[0][1],  # 1st element of 0th list in prediction list
            'bkl': prediction[0][2],
            'df': prediction[0][3],
            'mel': prediction[0][4],
            'nv': prediction[0][5],
            'vasc': prediction[0][6]  #
        }
    }
    return jsonify(response)  # return to front-end
