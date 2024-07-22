from PIL import Image, ImageOps
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import regex as re
import os


# folder path
dir_path = r'frame_and_mask_generation\\framesforlabels\\image5'

# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)


for index in res:

    img = np.zeros((960, 1280, 3), dtype = np.uint8)
    img = 255*img
    
    name = "frame_and_mask_generation/framesforlabels/label5/" + index
    print ('Creating ...' + name)
    cv2.imwrite(name, img)





                   
    