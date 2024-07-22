import cv2
import sys


import numpy as np
import tensorflow as tf
from tools.utils import *
from time import time
# Import the parser library
import argparse
from tools.metrics import dice_coef, iou, pxl_acc

# img size
image_size = 256
font = cv2.FONT_HERSHEY_SIMPLEX

parser = argparse.ArgumentParser()
parser.add_argument('--save', help='save video stream to output file', action='store_true')
parser.add_argument('--output', help='output file name', default='output.avi')
args = parser.parse_args()

# Open video capture
cap = cv2.VideoCapture(0)

# Load TensorFlow Lite model fsti-123

tflite_model_path = 'C:/Users/VermaPankhuri-Ferdin/Desktop/Work/smart-wipers-camera/weights/WeightsForAllModels/raindrops_segmentation_unet_mobilenetv2_tflite.tflite'


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_tensor = interpreter.get_input_details()
output_tensor = interpreter.get_output_details()

# Set video dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))

if args.save:
    # Set video codec and frame rate
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Other codecs also possible, e.g. XVID
    # Create VideoWriter object to write video frames
    out = cv2.VideoWriter(args.output, fourcc, 10, (int(cap.get(3)), int(cap.get(4))))



while cap.isOpened():
    # Read frame from video capture
    ret, frame = cap.read()
    if ret:
        

        h, w, _ = frame.shape

        if args.save:
            out.write(frame)

        ori_frame = frame

        # Convert to PIL Image
        frame = read_image_tflite(frame, image_size)
        input_data = np.array(frame, dtype=np.float32)
        interpreter.set_tensor(input_tensor[0]['index'], input_data)

        time_before = time()
        interpreter.invoke()
        time_after = time()

        output_data_tflite = interpreter.get_tensor(output_tensor[0]['index'])

        # Post-process output
        pred = output_data_tflite
        mask = mask_parse(pred, w, h)
        area_ratio = round(area_ratio_calculate(mask),2)
        #print(area_ratio)
        print(dice_coef(pred, mask))
        #print('Percentage: {:.2%}'.format(area_ratio / 100))

        res1 = video_overlay_creation(mask, ori_frame)

        # Put area ration in the video stream
        cv2.putText(res1, "Percentage:" , (0, 30), font, 0.5, (0, 0, 0))
        cv2.putText(res1, str(area_ratio), (100, 30), font, 0.5, (0, 0, 0))
        cv2.putText(res1, "%", (145, 30), font, 0.5, (0, 0, 0))

        # Display segmented frame
        cv2.imshow("Segmented Frame", res1)
        if cv2.waitKey(1) == ord("x") or cv2.getWindowProperty('Segmented Frame', 1) < 0:
                break

    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
