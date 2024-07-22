import os
import numpy as np
from tools.utils_preprocessing import read_image, read_mask
from tensorflow.keras.models import load_model
from tools.metrics import dice_coef, iou, pxl_acc
import statistics
from time import time


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
#test_dataset = tf_dataset(test_images, test_masks, batch=BATCH)

test_steps = (len(test_images)//BATCH)

if len(test_images) % BATCH != 0:
    test_steps += 1

# Loading Model
weights_address = '../weights/WeightsForAllModels/raindrops_segmentation_unet_vgg.h5'


model = load_model(weights_address, compile=False)  # tensor type

dc_avg = []
iou_avg = []
pa_avg = []
run_time = []
counter = 0

for i, (x, y) in enumerate(zip(test_images, test_masks)):
    x = read_image(x, IMAGE_SIZE)
    y = read_mask(y, IMAGE_SIZE)
    time_before = time()
    y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
    print(y.shape)
    print(y_pred.shape)
    time_after = time()
    pred_time = time_after-time_before

    run_time.append(pred_time)
    dc_avg.append(dice_coef(y, y_pred))
    iou_avg.append(iou(y, y_pred))
    pa_avg.append(pxl_acc(y,y_pred))
    counter = counter + 1
    print(counter)



# 'w' mode is used to write data. It will overwrite existing files.
with open('keras_unet_vgg_results.txt', 'w') as file:
    file.write("Runtime in ms:")
    file.write(str(statistics.mean(run_time)))
    file.write("\nDice Coefficient:")
    file.write(str(statistics.mean(dc_avg)))
    file.write("\nIOU:")
    file.write(str(statistics.mean(iou_avg)))
    file.write("\nPixel Accuracy:")
    file.write(str(statistics.mean(pa_avg)))


