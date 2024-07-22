import PIL as p
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model.unet_mobilenetv2 import unet_mobilenetv2
from  tools.utils_preprocessing import tf_dataset, load_data
from tools.metrics import dice_coef, dice_loss


from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K


input_dir = 'dataset/RaindropsOnWindshield/images'
mask_dir = 'dataset/RaindropsOnWindshield/masks'
PATH = 'dataset/RaindropsOnWindshield'

IMAGE_SIZE = 256
EPOCHS = 30
BATCH = 8
LR = 1e-4
smooth = 1e-15
num_class = 2
split=0.1



def main():
    (train_x, train_y), (valid_x, valid_y) = load_data(PATH, split, input_dir, mask_dir)

    train_dataset = tf_dataset(train_x, train_y, batch=BATCH)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)

    opt = tf.keras.optimizers.Nadam(LR)
    metrics = [dice_coef, Recall(), Precision()]
    model = unet_mobilenetv2(num_class, IMAGE_SIZE)
    model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
        ModelCheckpoint("weights/raindrops_segmentation_unet_mobilenetv2_new.h5", monitor='val_loss', verbose=0,
                        save_best_only=True, mode='min')
    ]

    train_steps = len(train_x) // BATCH
    valid_steps = len(valid_x) // BATCH

    if len(train_x) % BATCH != 0:
        train_steps += 1
    if len(valid_x) % BATCH != 0:
        valid_steps += 1

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
