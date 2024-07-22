from PIL import Image
import os
from PIL import Image

# Define the folder containing your images
folder_path = 'C:\\Users\\VermaPankhuri-Ferdin\\Downloads\\RaindropsOnWindshield_NewDataset2\\training\\masks\\F1\\'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
     print(filename)
    
     if filename.endswith(".png"):
        # Open the PNG image file
        png_image = Image.open(folder_path+filename)

        # Convert and save it as a JPEG image
        filename_without_extension, file_extension = os.path.splitext(filename)
        mask_name = filename_without_extension + ".jpg"
        png_image.save(folder_path+mask_name, "JPEG")
        print(png_image)
        # Close the image
        png_image.close()
        os.remove(folder_path+filename)

        print("Conversion completed.")

        
