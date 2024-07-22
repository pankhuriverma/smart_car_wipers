import os
import sys
sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
from tools.utils import predict_to_mask
from tools.utils import image_processing, video_overlay_creation, area_ratio_calculate
from tensorflow.keras.models import load_model


def main( ):

        # img size
        img_size = (256, 256)

        # keras model loading
        model = load_model(r"../weights/keras_model.h5")
        # model.summary()

        # frames reading
        cap = cv2.VideoCapture(0)

        while True:

                ret, frame = cap.read()

                if ret == False:

                    break

                h, w, _ = frame.shape

                ori_frame = frame

                # Convert to PIL Image
                frame_proc = image_processing(frame, img_size)

                # Keras model inference
                pred = model.predict(frame_proc, verbose=0)
                mask = predict_to_mask(pred, w, h)

                area_ratio = area_ratio_calculate(mask)
                print('Percentage: {:.2%}'.format(area_ratio / 100))

                # Generate overlay for the video stream
                res1 = video_overlay_creation(mask, ori_frame)

                # Put area ration in the video stream
                cv2.putText(res1, str(area_ratio), (30, 30), font, 1.2, (0, 0, 0))

                cv2.imshow("Segmented Frame", res1)
                if cv2.waitKey(1) == ord("x")  or cv2.getWindowProperty('Segmented Frame', 1) < 0:
                    break

            # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

        main()
        # parser = argparse.ArgumentParser()
        # # parser.add_argument("input_file", type=str)
        # # parser.add_argument("output_file", type=str)
        # # parser.add_argument("-v", "--verbose", action="store_true")
        # main(parser.parse_args())


