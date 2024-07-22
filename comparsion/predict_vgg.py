import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from time import time

from tools.utils import predict_to_mask, image_processing, area_ratio_calculate

img_size = (256, 256)
#frame rate
fps =30
num_classes = 2

