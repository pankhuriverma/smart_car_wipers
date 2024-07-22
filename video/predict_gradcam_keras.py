import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
from gradcam.gradcam_keras import ClassRoI, SegGradCAM
from gradcam.visualize_gradcam_keras import SegGradCAMplot
from model.unet_mobilenetv2 import mobilenetv2_model
from tools.utils import predict_to_mask, image_processing, area_ratio_calculate
tf.compat.v1.disable_eager_execution()

img_size = (256, 256)
# frame rate
fps = 30
num_classes = 2
font = cv2.FONT_HERSHEY_SIMPLEX

"""
Smart Wipers model inference code: based on UNet structure, and using GradCAM to generate 
the heatmap
"""

# The address in the cloud/server where the input video is stored.
# This video needs to be played on the widget.
video_address = r'../video/inputs/test_MediumRain_AVC.mp4'

# The address in the cloud/server where the file of the model weights and parameters is stored
weights_address = r'../weights/keras_raindrops_segmentation_unet_mobilenetv2_nightparkingscenario.h5'

frameCapture = cv2.VideoCapture(video_address)

# Load keras model
model = mobilenetv2_model(num_classes)
model = load_model(weights_address, compile=False)  # tensor type

while (True):
    # Read each frame
    ret, frame = frameCapture.read()

    # BGR converts to RGB
    # frame: the input image (Windshield) needs to be displayed on the widgets
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # Convert to PIL img for Seg Grad-CAM
    frame_gradcam = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert to PIL Images
    frame_proc = image_processing(frame,img_size)

    pred = model.predict(frame_proc)

    # predict array converts to mask
    # pred_mask: to predict mask of the input image also needs to be display when clicking
    # the button
    pred_mask = predict_to_mask(pred, w, h)

    # area_ratio: the percentage of the area of raindrops needs to be printed on the widget
    area_ratio = area_ratio_calculate(pred_mask)
    print('Percentage: {:.2%}'.format(area_ratio / 100))

    # mask visualize
    # plt.figure('mask')
    # plt.imshow(pred_mask)
    # plt.show()

    # Extracting the output of the last layer and the intermediate layer
    prop_from_layer = model.layers[-1].name
    prop_to_layer = model.layers[-2].name

    # GradCAM
    # The class 1 for raindrops, the class 0 for background
    cls = 1
    clsroi = ClassRoI(model, image=frame_proc, cls=cls)
    newsgc = SegGradCAM(model, frame_proc, cls, prop_to_layer, prop_from_layer, roi=clsroi,
                        normalize=True, abs_w=False, posit_w=False)
    newsgc.SGC()
    plotter = SegGradCAMplot(newsgc, model=model, image=frame_gradcam, n_classes=num_classes)
    plotter.explainClass()
