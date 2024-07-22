
import numpy as np
import cv2
from tools import utils as u
from tensorflow.keras.models import load_model

class SmartWiperModel:

    def __init__(self):
        self.image_size = 256
        self.weights_address = 'C:/Users/VermaPankhuri-Ferdin/Desktop/Work/smart-wipers-camera/weights/WeightsForAllModels/raindrops_segmentation_unet_mobilenetv2.h5'

    def predict(self):
        # Reading frames
        cap = cv2.VideoCapture(0)
        # Loading Model
        model = load_model(self.weights_address, compile=False)  # tensor type

        while cap.isOpened():

            ret, frame = cap.read()

            if ret:

                h, w, _ = frame.shape

                frame = u.read_image(frame, self.image_size)
                ori_frame = frame
                pred = model.predict(np.expand_dims(frame, axis=0))[0] > 0.5
                mask = u.mask_parse(pred, w, h)

                area_ratio = round(u.area_ratio_calculate(mask), 2)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        return area_ratio
