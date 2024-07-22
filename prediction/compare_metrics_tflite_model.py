import statistics
import sys
sys.path.append("../tools")
import numpy as np
import tensorflow as tf
from time import time
import utils as u
import os
import metrics as m

IMAGE_SIZE = 256
EPOCHS = 30
BATCH = 8
LR = 1e-4
smooth = 1e-15
num_class = 2
split = 0.1

test_input_dir = '../test_images/images'
test_masks_dir = '../test_images/masks'

test_images = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_input_dir) for f in files]
test_masks = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_masks_dir) for f in files]

test_steps = (len(test_images) // BATCH)

if len(test_images) % BATCH != 0:
    test_steps += 1

# Loading Model
weights_address = '../weights/WeightsForAllModels/raindrops_segmentation_unet_mobilenetv2_tflite.tflite'

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=weights_address)
interpreter.allocate_tensors()

# Get input and output details
input_tensor = interpreter.get_input_details()
output_tensor = interpreter.get_output_details()

dc_avg = []
iou_avg = []
pa_avg = []
run_time = []
counter = 0
h = 256
w = 256

for i, (x, y) in enumerate(zip(test_images[1190:], test_masks[1190:])):
    frame = u.read_image_tflite_from_filepath(x, IMAGE_SIZE)

    y = u.read_test_mask(y, IMAGE_SIZE)
    print(y.shape)
    input_data = np.array(frame, dtype=np.float32)
    interpreter.set_tensor(input_tensor[0]['index'], input_data)
    time_before = time()
    interpreter.invoke()
    time_after = time()
    pred_time = time_after - time_before
    pred = interpreter.get_tensor(output_tensor[0]['index']) > 0.5
    print(pred.shape)
    pred_mask = u.mask_parse(pred)
    print(pred_mask.shape)
    mask = np.mean(pred_mask, axis=2, keepdims=True)
    print(mask.shape)
    run_time.append(pred_time)
    dc_avg.append(m.dice_coef(y, mask))
    iou_avg.append(m.iou(y, mask))
    pa_avg.append(m.pxl_acc(y, mask))
    counter = counter + 1
    print(counter)

# 'w' mode is used to write data. It will overwrite existing files.
with open('final_tflite_unet_mobilenetv2_results.txt', 'w') as file:
    file.write("Runtime in ms:")
    file.write(str(statistics.mean(run_time)))
    file.write("\nDice Coefficient:")
    file.write(str(statistics.mean(dc_avg)))
    file.write("\nIOU:")
    file.write(str(statistics.mean(iou_avg)))
    file.write("\nPixel Accuracy:")
    file.write(str(statistics.mean(pa_avg)))

