import os
import sys
sys.path.append("../tools")
import numpy as np
import utils_preprocessing as up
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


IMAGE_SIZE = 256
EPOCHS = 30
BATCH = 8
LR = 1e-4
smooth = 1e-15
num_class = 2
split = 0.1

test_input_dir = '../test_images/images_for_paper'
test_masks_dir = '../test_images/masks'

test_images = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_input_dir) for f in files]
test_masks = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_masks_dir) for f in files]

# Loading Model
weights_address = '../weights/WeightsForAllModels/raindrops_segmentation_unet_mobilenetv2.h5'


model = load_model(weights_address, compile=False)  # tensor type

for i, (x) in enumerate(test_images):
    x = up.read_image(x, IMAGE_SIZE)


    #y = read_mask(y, IMAGE_SIZE)

    y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5


    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the first image on the first subplot

    axes[0].imshow(x)
    axes[0].set_title('Image')

    axes[1].imshow(y_pred)
    axes[1].set_title('Predicted Mask')



    # Display the images
    plt.show()



