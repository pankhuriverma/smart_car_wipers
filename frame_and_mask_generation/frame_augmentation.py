import cv2
import os
import Augmentor
import albumentations as A
import random

import cv2
from matplotlib import pyplot as plt
from PIL import Image

input_dir = 'C:/Users/VermaPankhuri-Ferdin/Downloads/RaindropsOnWindshield/images/v2'
mask_dir = 'C:/Users/VermaPankhuri-Ferdin/Downloads/RaindropsOnWindshield/masks/v2'

original_height = 1024
original_width = 1280

transform = A.Compose([
    
    A.RandomCrop(width=300, height=300),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(height=1024,width=1280),
    ])

   


images = [os.path.join(pth, f) for pth, dirs, files in os.walk(input_dir) for f in files]

masks = [os.path.join(pth, f) for pth, dirs, files in os.walk(mask_dir) for f in files]
index=1

for i, m in zip(images, masks):

    image = cv2.imread(i)
    mask = cv2.imread(m)
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    image_name = 'C:/Users/VermaPankhuri-Ferdin/Downloads/RaindropsOnWindshield/images/augv2/I' + str(index) + '.jpg'
    print ('Creating ...' + image_name)
    cv2.imwrite(image_name, transformed_image)
    name2 = 'C:/Users/VermaPankhuri-Ferdin/Downloads/RaindropsOnWindshield/masks/augv2/I' + str(index) + '.jpg'
    print ('Creating ...' + name2)
    cv2.imwrite(name2, transformed_mask)
    index = index + 1




"""p1 = Augmentor.Pipeline("C:\\Users\\VermaPankhuri-Ferdin\\Downloads\\RaindropsOnWindshield\\images\\1o")

p1.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p1.flip_left_right(probability=0.5)
p1.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p1.skew_tilt(probability=0.5,magnitude=1.0)
p1.sample(10)


p2 = Augmentor.Pipeline("C:\\Users\\VermaPankhuri-Ferdin\\Downloads\\RaindropsOnWindshield\\masks\\1o")

p2.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p2.flip_left_right(probability=0.5)
p2.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p2.skew_tilt(probability=0.5,magnitude=1.0)
p2.sample(10)
"""




 



    