import os

import sys

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#sys.path.append("../tools")

import numpy as np
import cv2
from tools import utils as u
from model.unet_mobilenetv2 import unet_mobilenetv2

from tensorflow.keras.models import load_model
#import keras_to_tflite_converter as conv
from time import time
# Import the parser library
import argparse

"""
UNet which takes mobilenetv2 as backbone
"""

parser = argparse.ArgumentParser()
parser.add_argument('--save', help='save video stream to output file', action='store_true')
parser.add_argument('--output', help='output file name', default='output.avi')
args = parser.parse_args()

# img size
image_size = 256
font = cv2.FONT_HERSHEY_SIMPLEX
# fps = 10 # Change this value to adjust frame rate



# Reading frames
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, fps)

# Set video dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))


    

# Loading Model
weights_address = 'C:/Users/VermaPankhuri-Ferdin/Desktop/Work/smart-wipers-camera/weights/WeightsForAllModels/raindrops_segmentation_unet_mobilenetv2.h5'


model = load_model(weights_address, compile=False)  # tensor type



if args.save:
            # Set video codec and frame rate
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Other codecs also possible, e.g. XVID
            # Create VideoWriter object to write video frames
            out = cv2.VideoWriter(args.output, fourcc, 10, (int(cap.get(3)), int(cap.get(4))))

# model.summary()

while cap.isOpened():

    ret, frame = cap.read()

    if ret:

        h, w, _ = frame.shape

        

        frame = u.read_image(frame, image_size)
        ori_frame = frame
        # pred = model.predict(frame, verbose=0)[0][:,:,-1]
        pred = model.predict(np.expand_dims(frame, axis=0))[0] > 0.5
        mask = u.mask_parse(pred, w, h)

        area_ratio = round(u.area_ratio_calculate(mask),2)
        print(area_ratio)
        #print('Percentage: {:.2%}'.format(area_ratio / 100))

        res1 = u.video_overlay_creation(mask, ori_frame)

        # Put area ration in the video stream
        cv2.putText(res1, "Percentage:" , (0, 30), font, 0.5, (0, 0, 0))
        cv2.putText(res1, str(area_ratio), (100, 30), font, 0.5, (0, 0, 0))
        cv2.putText(res1, "%", (145, 30), font, 0.5, (0, 0, 0))

        cv2.imshow("Segmented Frame", res1)
        if args.save:
            # Set video codec and frame rate
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Other codecs also possible, e.g. XVID
            # Create VideoWriter object to write video frames
            out = cv2.VideoWriter(args.output, fourcc, 10, (int(cap.get(3)), int(cap.get(4))))
            out.write(mask)

        if cv2.waitKey(1) == ord("x") or cv2.getWindowProperty('Segmented Frame', 1) < 0:
            break

    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
