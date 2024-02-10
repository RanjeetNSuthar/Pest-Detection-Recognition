# import pyrebase
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np

import sys
import collections
from collections.abc import MutableMapping

# firebaseConfig = {
#     "apiKey": "",
#     "authDomain": "",
#     "databaseURL": "",
#     "projectId": "",
#     "storageBucket": "",
#     "messagingSenderId": "",
#     "appId": "",
#     "measurementId": "",
#     "serviceAccount": "",
#     "databaseURL": ""
# }

# firebase = pyrebase.initialize_app(firebaseConfig)
# storage = firebase.storage()
# db = firebase.database()

# image_name = db.child("latest_search").child("image_url").get()
# storage.download(image_name, "image")

X_test = pd.DataFrame({"image_name": [
                      r"<localpath>\Driver\image2.jpeg"]})["image_name"].to_numpy()

# define a function to preprocess the images and turn them into tensors
trained_model = tf.keras.models.load_model(
    r'<localpath>\Driver\my_model1.hdf5', custom_objects={'KerasLayer': hub.KerasLayer})

IMG_SIZE = 224
BATCH_SIZE = 32


def process_image(image_path, size=IMG_SIZE):
    '''
    takes image file path as an input and returns a tensor of specified size = (size x size)
    '''

    # takes input image and convert it into tensor object
    image = tf.io.read_file(image_path)
    # Turn the jpeg tensor image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # convert the colour channel value from (0-255) to (0-1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize image as (size X size)
    image = tf.image.resize(image, size=[size, size])

    return image


def get_data_batches(x, batch_size=BATCH_SIZE):
    """
    Creates batches of data out of image(image paths) and accepts test data as input (no labels).
    """

    print("Creating test data batches")
    dataset = tf.data.Dataset.from_tensor_slices(x)
    data_batch = dataset.map(process_image).batch(batch_size)
    return data_batch


test_data_batches = get_data_batches(X_test)
pest_type = pd.read_csv(
    r"<localpath>\Driver\classes.csv", names=["label", "type"])["type"]

predictions = trained_model.predict(test_data_batches, verbose=1)

# Turn prediction probabilities into their respective label (easier to understand)


def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return pest_type[np.argmax(prediction_probabilities)]


pred_label = get_pred_label(predictions[0])
print(f"Pest Category : {pred_label}")
print(f"Confidence : {np.max(predictions[0])*100}")
