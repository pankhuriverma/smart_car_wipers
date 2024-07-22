from PIL import Image, ImageOps
import cv2
import numpy as np
from tensorflow import keras
import os
#from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
"""test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '../test_images/images_for_paper',
    target_size=(64, 64),
    batch_size=2,
    class_mode=None,
    shuffle=False
)
"""

def dataset_load(input_dir, mask_dir, split = 0.1):
    images = [os.path.join(pth, f) for pth, dirs, files in os.walk(input_dir) for f in files]
    masks = [os.path.join(pth, f) for pth, dirs, files in os.walk(mask_dir) for f in files]

    total_size = len(images)

    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    return x

def read_mask(path,IMAGE_SIZE):

    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    y.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


def image_processing(frame, img_size):
    frame_arr = Image.fromarray(np.uint8(frame))
    new_img = frame_arr.resize(img_size, Image.ANTIALIAS)

    img_tf = np.array(new_img)
    new_img = img_tf / 255.0

    new_img = np.expand_dims(new_img, 0)
    new_img = np.array(new_img, dtype=np.float32)

    return new_img


def predict_to_mask(pred_array, width, height):
    # print("pred arr")
    # print(pred_array.shape)
    mask = np.argmax(pred_array, axis=-1)
    # print("mask arg max")
    # print(mask.shape)
    mask = np.squeeze(mask)
    # print("mask squeeze")
    # print(mask.shape)
    mask = np.expand_dims(mask, axis=-1)
    # print("mask expand dim")
    # print(mask.shape)
    mask = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    mask = mask.convert("RGB")
    mask_reshape = mask.resize((width, height))

    return mask_reshape


def area_ratio_calculate(pred_mask):
    img = cv2.cvtColor(np.asarray(pred_mask), cv2.COLOR_RGB2GRAY)

    # countNoneZero: compute non-zero pixel values, the input must be one-channel
    ratio_white = cv2.countNonZero(img) / (img.size)
    colorPercent = ratio_white * 100

    return colorPercent


def video_overlay_creation(mask, ori_frame):
    mask_opencv = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(mask_opencv.astype(np.uint8), 0.5, 255, 0)
    contours, im = cv2.findContours(mask_opencv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
    res1 = cv2.drawContours(ori_frame.copy(), contours=contours, contourIdx=-1, color=(64, 224, 208), thickness=1)

    return res1


def read_image(frame, img_size):
    x = cv2.resize(frame, (img_size, img_size))
    x = x / 255.0
    return x


def read_image_tflite(x, img_size):

    x = cv2.resize(x, (img_size, img_size))
    x = x / 255.0
    new_img = np.expand_dims(x, 0)
    return new_img


def read_image_tflite_from_filepath(x, img_size):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (img_size, img_size))
    x = x / 255.0

    new_img = np.expand_dims(x, 0)

    return new_img


def read_test_image(frame, img_size):
    x = cv2.imread(frame, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (img_size, img_size))
    x = x / 255.0

    return x


def read_test_mask(frame, img_size):
    x = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (img_size, img_size))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)

    return x


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask



def mask_parse_tflite(mask, width, height):
    mask = np.squeeze(mask)

    mask = [mask, mask, mask]

    mask = np.transpose(mask, (1, 2, 0))

    mask = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    mask = mask.convert("RGB")
    mask_reshape = mask.resize((width, height))

    return mask_reshape

def preprocess_image(image, target_size):
    #image = load_img(image, target_size=target_size)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # rescale
    return image

def preprocess_image_classification(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    #image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # rescale
    return image





