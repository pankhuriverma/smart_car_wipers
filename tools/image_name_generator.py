import os

# Set the path to your images folder
image_folder_path = '../train_classify_images/train_classify_images/rain_new'
new_folder_path = '../train_classify_images/train_classify_images/rain'
file_extension = '.png'  # Change this to the appropriate file extension (e.g., .png, .jpeg)

# Get a list of all files in the directory
files = os.listdir(image_folder_path)
print(files)

# Counter for the image numbering
counter = 1

# Loop through the files and rename them
for file in files:
    # Construct the old file path
    old_file_path = os.path.join(image_folder_path, file)

    # Check if it's a file and not a directory
    if os.path.isfile(old_file_path):
        # Construct the new file path
        new_file_name = f'rain_{counter}{file_extension}'
        new_file_path = os.path.join(new_folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)

        # Increment the counter
        counter += 1

print(f"Renamed {counter - 1} files.")
