import cv2
import sys
sys.path.append("..")
from tools.utils import predict_to_mask
from tools.utils import image_processing, video_overlay_creation, area_ratio_calculate
import tensorflow as tf


# img size
img_size = (256, 256)
font = cv2.FONT_HERSHEY_SIMPLEX

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=r"../weights/tflite_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_tensor = interpreter.get_input_details()
output_tensor = interpreter.get_output_details()

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
    frame_proc = image_processing(frame, img_size)

    # TFlite model inference
    interpreter.set_tensor(input_tensor[0]['index'], frame_proc)
    interpreter.invoke()
    output_data_tflite = interpreter.get_tensor(output_tensor[0]['index'])

    # Post-process output
    mask = predict_to_mask(output_data_tflite, w, h)
    area_ratio = area_ratio_calculate(mask)
    res1 = video_overlay_creation(mask, ori_frame)

    # Put area ratio in the video stream
    cv2.putText(res1, str(area_ratio), (30, 30), font, 1.2, (0, 0, 0))

    # Display segmented frame
    cv2.imshow("Segmented Frame", res1)
    if cv2.waitKey(1) == ord("x") or cv2.getWindowProperty('Segmented Frame', 1) < 0:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
