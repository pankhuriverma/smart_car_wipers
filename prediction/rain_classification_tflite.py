import os
import sys
sys.path.append("../tools")
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import utils as u
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import time
tf.compat.v1.disable_eager_execution()

IMAGE_SIZE = 256
classification_weights_address = '../weights/WeightsForAllModels/image_classification_tflite.tflite'
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=classification_weights_address)
interpreter.allocate_tensors()

# Get input and output details
input_tensor = interpreter.get_input_details()
output_tensor = interpreter.get_output_details()
test_image_dir = '../test_images/images_for_paper'
test_images = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_image_dir) for f in files]

for i, (x) in enumerate(test_images):
    frame = u.read_image_tflite_from_filepath(x, IMAGE_SIZE)
    input_data = np.array(frame, dtype=np.float32)
    frame = np.squeeze(input_data)
    h, w, _ = frame.shape
    print(frame.shape)
    interpreter.set_tensor(input_tensor[0]['index'], input_data)
    time_before = time.time()
    interpreter.invoke()
    time_after = time.time()
    pred_time = time_after - time_before
    pred = interpreter.get_tensor(output_tensor[0]['index']) > 0.5
    #y_pred = u.mask_parse(pred)
    print(pred)
    print(pred_time)




