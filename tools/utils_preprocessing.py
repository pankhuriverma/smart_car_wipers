
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def read_image(path, IMAGE_SIZE):

    x = cv2.imread(path, cv2.IMREAD_COLOR)
    print('image shape')

    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    print(x.shape)
    return x

def read_mask(path, IMAGE_SIZE):

    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print('mask shape')

    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    print(x.shape)
    return x




def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask



def read_and_rgb(x):
    x = cv2.imread(x)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x



def tf_parse(x, y, IMAGE_SIZE):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    y.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    return x, y


def tf_dataset(x, y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


def load_data(path, split, input_dir, mask_dir):
    
    
    images = [os.path.join(pth, f) for pth, dirs, files in os.walk(input_dir) for f in files]
    masks = [os.path.join(pth, f) for pth, dirs, files in os.walk(mask_dir) for f in files]

    total_size = len(images)
    
    valid_size = int(split * total_size)
    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)



