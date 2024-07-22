import os
import sys
sys.path.append("../tools")
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import utils as u

"""classification_weights_address = '../weights/WeightsForAllModels/image_classification.h5'
classification_model = load_model(classification_weights_address, compile=False)  # tensor type
test_image_dir = '../test_images/images'
test_images = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_image_dir) for f in files]
preprocessed_image = u.preprocess_image_classification(test_images[1], target_size=(256, 256))
output = classification_model.predict(preprocessed_image)
print(output)
predicted_class = (output > 0.5).astype('int32')
print(predicted_class)"""
"""if predicted_class[0][0] == 0:
    print("No Rain")
else:
    print("Rain")
"""



from tensorflow.keras.models import load_model
classification_weights_address = '../weights/WeightsForAllModels/image_classification_largemodel_epochs20.h5'
model = load_model(classification_weights_address, compile=False)  # tensor type
#test_image_dir = '../test_images/images'
test_image_dir = '../test_images/images'
test_images = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_image_dir) for f in files]
counter = 0
output_list = []
for image in test_images[1180:]:
    preprocessed_image = u.preprocess_image_classification(image, target_size=(256, 256))
    output = model.predict(preprocessed_image)
    print(output)
    predicted_class = (output > 0.5).astype('int32')
    output_list.append(predicted_class[0][0])
    print(predicted_class[0][0])
    counter = counter + 1
    print(counter)

print(output_list)








