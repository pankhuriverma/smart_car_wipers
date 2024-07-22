import sys
import os
sys.path.append("../tools")
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import utils as u



tf.compat.v1.disable_eager_execution()

img_size = 256
# frame rate
fps = 30
num_classes = 1

"""
Smart Wipers model inference code: based on UNet structure, and using GradCAM to generate 
the heatmap
"""

# The address in the cloud/server where the input video is stored.
# This video needs to be played on the widget.

video_address = '../video/inputs/test_SlightRain_AVC.mp4'

# The address in the cloud/server where the file of the model weights and parameters is stored
weights_address = '../weights/WeightsForAllModels/raindrops_segmentation_unet_mobilenetv2.h5'
classification_weights_address = '../weights/WeightsForAllModels/image_classification_largemodel_epochs20.h5'
classification_model = load_model(classification_weights_address, compile=False)
frameCapture = cv2.VideoCapture(video_address)



# Load keras model

model_unet = load_model(weights_address, compile=False)  # tensor type

while (True):
    # Read each frame
    ret, frame = frameCapture.read()
    h, w, _ = frame.shape
    preprocessed_image = u.preprocess_image(frame, target_size=(256, 256))
    output = classification_model.predict(preprocessed_image)
    predicted_class = (output > 0.5).astype('int32')
    print(output)
    if predicted_class[0][0] == 0:
        print("No Rain")
    else:
        print("Rain")

        pred = model_unet.predict(preprocessed_image)[0] > 0.5

        pred_mask = u.mask_parse(pred)
        input_img = np.squeeze(preprocessed_image)
        plt.figure(figsize=(10,10))
        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        plt.imshow(pred)
        plt.title('Image Mask')

        plt.show(block=False)
        plt.pause(3)
        plt.close("all")

