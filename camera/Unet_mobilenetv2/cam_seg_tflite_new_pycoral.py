#import tensorflow as tf
import cv2
import numpy as np
#import tensorflow as tf
from tools.utils import read_image_tflite, mask_parse
import sys
sys.path.append("..")
from time import time

from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.pybind._pywrap_coral import SetVerbosity as set_verbosity
set_verbosity(10)

# img size
image_size = 256

# Load TensorFlow Lite model
#interpreter = tf.lite.Interpreter(model_path="weights/tflite_new_model.tflite")
#interpreter = tflite.Interpreter(model_path = "weights/tflite_new_model.tflite" , 
#experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
#interpreter.allocate_tensors()

# Get input and output details
#input_tensor = interpreter.get_input_details()
#output_tensor = interpreter.get_output_details()

# Initialize the TF pycoral interpreter 
interpreter = edgetpu.make_interpreter(r"../weights/tflite_new_model.tflite")
interpreter.allocate_tensors()





# Open video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    ori_frame = frame

    # Convert to PIL Image
    #frame = Image.fromarray(np.uint8(frame))
    #frame_proc = image_processing(frame_arr,img_size)
    #input_data = np.array(frame_proc, dtype=np.float32)
    frame = read_image_tflite(frame, image_size)
    input_data = np.array(frame, dtype=np.float32)
    #interpreter.set_tensor(input_tensor[0]['index'], input_data)

    #pycoral
    # Run an inference
    common.set_input(interpreter, input_data)

    # # Pre-process frame
    # frame = cv2.resize(frame, (256, 256))
    # frame = frame.astype("float32") / 255
    #
    # # Run TensorFlow Lite model
    # input_tensor().resize((1, 256, 256, 3))
    # input_tensor()[0, :, :, :] = frame
    time_before = time()
    interpreter.invoke()
    time_after = time()

    total_tflite_time = time_after - time_before
    print("Total prediction time for tflite without opt model is: ", total_tflite_time)

    

    #output_data_tflite = interpreter.get_tensor(output_tensor[0]['index'])
    #pycoral output
    pred = classify.get_classes(interpreter, top_k=1)

    # Post-process output
    #pred = output_data_tflite
    # output = cv2.resize(output, (frame.shape[1], frame.shape[0]))
    # output =output.astype(np.float32)
    # output = output.argmax(axis=-1)
    # output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    # output = cv2.applyColorMap(output)
    #mask = predict_to_mask(pred, w, h)

    
    pred_mask = mask_parse(pred, w, h)
    mask_opencv = cv2.cvtColor(np.asarray(pred_mask), cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(mask_opencv.astype(np.uint8), 0.5, 255, 0)
    contours, im = cv2.findContours(mask_opencv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
    res1 = cv2.drawContours(ori_frame.copy(), contours=contours, contourIdx=-1, color=(64, 224, 208), thickness=1)

    # Display segmented frame
    cv2.imshow("Segmented Frame", res1)
    if cv2.waitKey(1) == ord("x") or cv2.getWindowProperty('Segmented Frame', 1) < 0:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
