import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from time import time
# import pixellib
from tools.utils import predict_to_mask, image_processing, area_ratio_calculate

tf.compat.v1.disable_eager_execution()

img_size = (256, 256)
# frame rate
fps = 30
num_classes = 2

"""
The address in the cloud/server where the input video is stored. This video needs to be played on the widget.
"""
video_address = r'../video/inputs/test_MediumRain_AVC.mp4'

frameCapture = cv2.VideoCapture(video_address)
tflite_model_path = r"../weights/raindrops_segmentation_unet_mobilenetv2_dicelos_final.tflite"

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()  # list type
output_details = interpreter.get_output_details()

# Test the model on input data.
input_shape = input_details[0]['shape']

while (True):
    # Read each frame
    ref, frame = frameCapture.read()

    # BGR converts to RGB
    # frame: the input image (Windshield) needs to be displayed on the widgets
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # convert to PIL img for Seg Grad-CAM
    frame_gradcam = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_frame_gradcam = np.array(frame_gradcam, dtype=np.float32)
    # input image visualize
    # plt.figure('input')
    # plt.imshow(frame)
    # plt.show()

    # Convert to PIL Image
    frame_proc = image_processing(frame,img_size)

    interpreter.set_tensor(input_details[0]['index'], frame_proc)
    # print("input data tensor")
    # print(input_details)

    time_before = time()
    interpreter.invoke()
    time_after = time()

    total_tflite_time = time_after - time_before
    print("Total prediction time for tflite without opt model is: ", total_tflite_time)

    output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
    # print("output data tensor")
    # print(output_data_tflite)

    # print("The tflite w/o opt prediction for this image is: ", output_data_tflite, " 0=Uninfected, 1=Parasited")
    pred = output_data_tflite

    # predict array converts to mask
    """
    pred_mask: the predict mask of the input image also needs to be display when clicking the button
    (temporary substitution of heatmap, the heatmap is still under construction )
    """
    pred_mask = predict_to_mask(pred, w, h)

    # print("output data tensor")
    # print(pred_mask)
    # print("##########get ops details")
    # print(interpreter._get_ops_details())
    # area ratio calculate
    """
    area_ratio: the percentage of the area of raindrops needs to be printed on the widget
    """
    area_ratio = area_ratio_calculate(pred_mask)
    print('Percentage: {:.2%}'.format(area_ratio / 100))

    # mask visualize
    # plt.figure('mask')
    # plt.imshow(pred_mask)
    # plt.show()
    
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask)
    plt.title('Image Mask')

    plt.show(block=False)
    plt.pause(3)
    plt.close("all")
   

"""
TensorFlow 2.4 doesn't support read the output from middle layer. Later try with the latest version TensorFlow 2.9
"""
# prop_from_layer = model.layers[-1].name
# prop_to_layer = model.layers[-2].name
# prop_to_layer = 'block5_conv3'
# prop_from_layer = 172
# prop_to_layer = 170


# The class 1 for raindrops, the class 0 for background
# cls = 1
# clsroi = ClassRoI(interpreter, image=input_data, cls=cls)
# newsgc = SegGradCAM(interpreter, input_data, pred_mask, cls, prop_to_layer, prop_from_layer, roi=clsroi, normalize=True, abs_w=False,
# posit_w=False)
# newsgc.SGC()
# plotter = SegGradCAMplot(newsgc, model=interpreter, image=input_frame_gradcam , n_classes=num_classes)
# plotter.explainClass()
