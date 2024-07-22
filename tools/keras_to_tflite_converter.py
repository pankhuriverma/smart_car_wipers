import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.python.keras import backend as K
import keras.losses
"""
LR = 1e-4
smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)"""




model = tf.keras.models.load_model('../train_classify_images_all/image_classification_largemodel_epochs20.h5', compile = False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
converter = tf.lite.TFLiteConverter.from_keras_model(model)





# optimisation
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# convert to TFlite
tflite_model = converter.convert()

with open('../weights/WeightsForAllModels/image_classification_largemodel_epochs20_tflite.tflite', 'wb') as f:
  f.write(tflite_model)

print("Model converted")