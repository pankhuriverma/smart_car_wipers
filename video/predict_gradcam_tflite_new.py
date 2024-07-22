import sys
import os
sys.path.append("../tools")


import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import time
# import pixellib

import utils as u
import statistics

tf.compat.v1.disable_eager_execution()


image_size = 256

"""
The address in the cloud/server where the input video is stored. This video needs to be played on the widget.
"""
video_address = '../video/inputs/test_SlightRain_AVC.mp4'

frameCapture = cv2.VideoCapture(video_address)
classification_tflite_model = '../weights/WeightsForAllModels/image_classification_largemodel_epochs20_tflite.tflite'
tflite_model_path = "../weights/WeightsForAllModels/raindrops_segmentation_unet_mobilenetv2_tflite.tflite"

# Load the TFLite classify model and allocate tensors.
interpreter_classify = tf.lite.Interpreter(model_path=classification_tflite_model)
interpreter_classify.allocate_tensors()

# Get input and output details
classify_input_tensor = interpreter_classify.get_input_details()
classify_output_tensor = interpreter_classify.get_output_details()


# Load the TFLite unet model and allocate tensors.
interpreter_unet = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter_unet.allocate_tensors()

# Get input and output details
unet_input_tensor = interpreter_unet.get_input_details()
unet_output_tensor = interpreter_unet.get_output_details()


# Test the model on input data.
input_shape = classify_input_tensor[0]['shape']


classify_avg_pred_time = []
segment_avg_pred_time = []
total_avg_pred_time = []
print(frameCapture)
while frameCapture.isOpened():
    # Read each frame
    ref, frame = frameCapture.read()
    if not ref:
        break  # Exit loop if no frame is read
    h, w, _ = frame.shape
    # Convert to PIL Image
    frame_proc = u.read_image_tflite(frame, image_size)

    input_data = np.array(frame_proc, dtype=np.float32)
    #frame_img = np.squeeze(input_data)
    interpreter_classify.set_tensor(classify_input_tensor[0]['index'], input_data)
    time_before_total = time.time()
    time_before_classify_model = time.time()
    interpreter_classify.invoke()
    time_after_classify_model = time.time()
    pred_time_classify = time_after_classify_model - time_before_classify_model
    print("Inference time of classification model: ", pred_time_classify)
    classify_avg_pred_time.append(pred_time_classify)
    pred = interpreter_classify.get_tensor(classify_output_tensor[0]['index']) > 0.5


    if pred[0] == False:
        print("No Rain")
    else:
        print("Rain")



        interpreter_unet.set_tensor(unet_input_tensor[0]['index'], input_data)


        time_before_unet_model = time.time()
        interpreter_unet.invoke()
        time_after_unet_model = time.time()

        pred_time_unet = time_after_unet_model - time_before_unet_model
        time_after_total = time.time()
        total_pred_time = time_after_total - time_before_total
        print("Inference time of unet model: ", pred_time_unet)
        segment_avg_pred_time.append(pred_time_unet)
        print("Total inference time of the model: ", total_pred_time)
        total_avg_pred_time.append(total_pred_time)

        output_data_tflite = interpreter_unet.get_tensor(unet_output_tensor[0]['index'])

        pred = output_data_tflite
        pred_mask = u.mask_parse_tflite(pred, w, h)

        """
        area_ratio: the percentage of the area of raindrops needs to be printed on the widget
        """
        area_ratio = u.area_ratio_calculate(pred_mask)
        print('Percentage: {:.2%}'.format(area_ratio / 100))

        """        plt.figure(figsize=(20, 20))
        plt.subplot(1, 2, 1)
        plt.imshow(frame)
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask)
        plt.title('Image Mask')

        plt.show(block=False)
        plt.pause(3)
        plt.close("all")"""

frameCapture.release()


# 'w' mode is used to write data. It will overwrite existing files.
with open('two_stage_model_inference_results.txt', 'w') as file:
    file.write("Classification model average runtime in ms:")
    file.write(str(statistics.mean(classify_avg_pred_time)))
    file.write("\nSegmentation model average runtime in ms:")
    file.write(str(statistics.mean(segment_avg_pred_time)))
    file.write("\nCombined model average runtime in ms:")
    file.write(str(statistics.mean(total_avg_pred_time)))




