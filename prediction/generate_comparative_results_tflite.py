import os
import sys
sys.path.append("../tools")
import numpy as np
import utils as u
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import time


IMAGE_SIZE = 256
EPOCHS = 30
BATCH = 8
LR = 1e-4
smooth = 1e-15
num_class = 2
split = 0.1

test_input_dir = '../test_images/images_for_paper'


test_images = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_input_dir) for f in files]


# Loading Model
weights_address = '../weights/WeightsForAllModels/raindrops_segmentation_unet_mobilenetv2_tflite.tflite'

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=weights_address)
interpreter.allocate_tensors()

# Get input and output details
input_tensor = interpreter.get_input_details()
output_tensor = interpreter.get_output_details()

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
    y_pred = u.mask_parse_tflite(pred, w, h)





    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the first image on the first subplot

    axes[0].imshow(frame)
    axes[0].set_title('Image')

    axes[1].imshow(y_pred)
    axes[1].set_title('Predicted Mask')



    # Display the images
    plt.show()



