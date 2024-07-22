
#import cv2
import os
import random
import shutil
import re

input_dir = 'C:\\Users\\VermaPankhuri-Ferdin\\Downloads\\RaindropsOnWindshield_NewDataset2\\training\\images\\'
mask_dir = 'C:\\Users\\VermaPankhuri-Ferdin\\Downloads\\RaindropsOnWindshield_NewDataset2\\training\\masks\\'
test_image_dir='C:\\Users\\VermaPankhuri-Ferdin\\Downloads\\RaindropsOnWindshield_NewDataset2\\test\\images'
test_mask_dir='C:\\Users\\VermaPankhuri-Ferdin\\Downloads\\RaindropsOnWindshield_NewDataset2\\test\\masks'

#images = [os.path.join(pth, f) for pth, dirs, files in os.walk(input_dir) for f in files]
#masks = [os.path.join(pth, f) for pth, dirs, files in os.walk(mask_dir) for f in files]

input_image_folders=[]
input_mask_folders=[]

folder_list_level1 = ['1o','2o','8o','D1','D2','F1','v2']
folder_list_level2_D3 = ['D3\\D3_0', 'D3\\D3_1', 'D3\\D3_2'] 
folder_list_level2_D4 = ['D4\\D4_0', 'D4\\D4_1']

#to get all folder names on level 1
for name in folder_list_level1:
    input_image_folders.append(input_dir + name)
    input_mask_folders.append(mask_dir + name)

#to get all folder names on level 2 for D3 folder
for name in folder_list_level2_D3:
    input_image_folders.append(input_dir + name)
    input_mask_folders.append(mask_dir + name)

#to get all folder names on level 1 for D4 folder
for name in folder_list_level2_D4:
    input_image_folders.append(input_dir + name)
    input_mask_folders.append(mask_dir + name)

print(input_image_folders)
print(input_mask_folders)



for input_folder, mask_folder  in zip(input_image_folders,input_mask_folders):

    # List of input images
    image_filenames = os.listdir(input_folder)
   

    # Randomly select 100 images
    selected_image_filenames = random.sample(image_filenames, 100)
  

    # Iterate through the selected images
    for filename in selected_image_filenames:
        
        input_image = os.path.join(input_folder, filename)
        input_mask = os.path.join(mask_folder, filename)
        
        # Create the paths for the test folders
        test_image = os.path.join(test_image_dir, filename)
        test_mask = os.path.join(test_mask_dir, filename)
        
        # Move the images 
        shutil.move(input_image, test_image)
        shutil.move(input_mask, test_mask)

        # Delete the moved images
        #os.remove(input_image)
        #os.remove(input_mask)

