import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tensorflow.keras import layers, models
image_directory = '../train_classify_images_all/train_classify_images'
# Extract filenames and labels
filenames = os.listdir(image_directory)

for i in filenames:
    i.replace(i[0:65],'')
labels = ['rain' if image.startswith('r') else 'no_rain' for image in filenames]

df = pd.DataFrame({
    'filename': filenames,
    'label': labels
})
print(df['label'])

# Splitting dataset into training and validation
train_df = df.sample(frac=0.8, random_state=42)
validation_df = df.drop(train_df.index)

# Image generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_directory,
    x_col='filename',
    y_col='label',
    target_size=(256,256),
    class_mode='binary',
    batch_size=32
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=image_directory,
    x_col='filename',
    y_col='label',
    target_size=(256,256),
    class_mode='binary',
    batch_size=32
)


def build_model(num_classes):
    model = models.Sequential()
    # Adjusted input shape to (256, 256, 3)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))  # Added an extra max pooling layer
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    return model


model = build_model(num_classes=1)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // 32,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_df) // 32
)

model.save('image_classification_largeCNNmodel_epochs20_sigmoid.h5')

def preprocess_image_classification(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    #image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # rescale
    return image


classification_weights_address = 'image_classification_largeCNNmodel_epochs20_sigmoid.h5'
classification_model = load_model(classification_weights_address, compile=False)  # tensor type"""
test_image_dir = '../test_images/images_for_paper'
counter = 0
test_images = [os.path.join(pth, f) for pth, dirs, files in os.walk(test_image_dir) for f in files]
for image in test_images:
    preprocessed_image = preprocess_image_classification(test_images, target_size=(256, 256))
    output = classification_model.predict(preprocessed_image)
    print(output)
    predicted_class = (output > 0.5).astype('int32')
    print(predicted_class[0][0])
    counter = counter + 1
    print(counter)