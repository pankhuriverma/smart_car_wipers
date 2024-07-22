import numpy as np
from tensorflow.keras import backend as K
import cv2
from skimage import measure
import tensorflow as tf

"""
GradCAM applicable to tflite model
"""

#https://github.com/kiraving/SegGradCAM
class SuperRoI:  # or rename it to ClassRoI
    def __init__(self, image =None):
        self.image = image
        self.roi = 1
        self.fullroi = None
        self.i = None
        self.j = None

    def setRoIij(self):
        #print("Shape of RoI: ", self.roi.shape)
        self.i = np.where(self.roi == 1)[0]
        self.j = np.where(self.roi == 1)[1]
        #print("Lengths of i and j index lists:", len(self.i), len(self.j))

    def meshgrid(self):
        # mesh for contour
        ylist = np.linspace(0, self.image.shape[2], self.image.shape[2])
        xlist = np.linspace(0, self.image.shape[1], self.image.shape[1])
        return np.meshgrid(xlist, ylist) #returns X,Y

class ClassRoI(SuperRoI):
    def __init__(self, interpreter, image, cls):
        # preds = model.predict(image)[0]
        # preds = model.predict(image,steps = 1)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

       
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
        preds = output_data_tflite
        max_preds = np.argmax(preds, axis=-1)
        self.image = image
        self.roi = np.round(preds[..., cls] * (max_preds == cls)).reshape(image.shape[-3], image.shape[-2])
        self.fullroi = self.roi
        self.setRoIij()

    def connectedComponents(self):
        all_labels = measure.label(self.fullroi, background=0)
        (values, counts) = np.unique(all_labels * (all_labels != 0), return_counts=True)
        print("connectedComponents values, counts: ", values, counts)
        return all_labels, values, counts

    def largestComponent(self):
        all_labels, values, counts = self.connectedComponents()
        # find the largest component
        ind = np.argmax(counts[values != 0]) + 1  # +1 because indexing starts from 0 for the background
        print("argmax: ", ind)
        # define RoI
        self.roi = (all_labels == ind).astype(int)
        self.setRoIij()

    def smallestComponent(self):
        all_labels, values, counts = self.connectedComponents()
        ind = np.argmin(counts[values != 0]) + 1
        print("argmin: ", ind)  #
        self.roi = (all_labels == ind).astype(int)
        self.setRoIij()

class SegGradCAM:
    """Seg-Grad-CAM method for explanations of predicted segmentation masks.
    Seg-Grad-CAM is applied locally to produce heatmaps showing the relevance of a set of pixels
    or an individual pixel for semantic segmentation.
    """

    def __init__(self, interpreter, image, pred_mask,  cls=-1, prop_to_layer='activation_9', prop_from_layer='last',
                 roi=SuperRoI(),  # 1, #default: explain all the pixels that belong to cls
                 normalize=True, abs_w=False, posit_w=False):



        self.input_model = interpreter
        self.image = image # the image after pre-processing
        self.mask = pred_mask
        # if cls == None:
        # TODO: add option cls=-1 (predicted class) and cls=None (gt class)
        # TODO print model's confidence (probability) in prediction
        self.cls = cls  # class
        # prop_from_layer is the layer with logits prior to the last activation function
        
       
        self.prop_from_layer = prop_from_layer
        self.prop_to_layer = prop_to_layer  # an intermediate layer, typically of the bottleneck layers

        self.roi = roi  # M, a set of pixel indices of interest in the output mask.
        self.normalize = normalize  # [True, False] normalize the saliency map L_c
        self.abs_w = abs_w  # if True, absolute function is applied to alpha_c
        self.posit_w = posit_w  # if True, ReLU is applied to alpha_c

        self.alpha_c = None  # alpha_c, weights for importance of feature maps
        self.A = None  # A, feature maps from the intermediate prop_to_layer
        self.grads_val = None  # gradients of the logits y with respect to all pixels of each feature map 𝐴^𝑘
        self.cam = None  # activation map L_c

        self.cam_max = None

    def featureMapsGradients(self):

        """ This method corresponds to the formula:
        Sum [(d Sum y^c_ij) / (d A^k_uv)] , where
        y^c_ij are logits for every pixel 𝑥_𝑖𝑗 and class c. Pixels x_ij are defined by the region of interest M.
        A^k is a feature map number k. u,v - indexes of pixels of 𝐴^𝑘.
        Return: A, gradients of the logits y with respect to all pixels of each feature map 𝐴^𝑘
        """
        preprocessed_input = self.image
        self.input_model.allocate_tensors()

        # Get input and output tensors.
        input_details = self.input_model.get_input_details()
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(input_details)
        output_details = self.input_model.get_output_details()

       
        self.input_model.set_tensor(input_details[0]['index'], preprocessed_input)

       
        self.input_model.invoke()
       
        #y_c = self.input_model.get_layer(self.prop_from_layer).output[
        #          ..., self.cls] * self.roi.roi  # Mask the region of interest
        # print("y_c: ", type(y_c), np.array(y_c))
        #conv_output = self.input_model.get_layer(self.prop_to_layer).output
        # print("conv_output: ", type(conv_output), np.array(conv_output))
        y_c = self.input_model.get_tensor(72)
        print("y_c")
        print(y_c)
        conv_output = self.input_model.get_tensor(70)
        grads = K.gradients(y_c, conv_output)[0]
        # print("grads: ", type(grads), grads)

        # TF2.0 GradientTape for eager mode, seems not work
        # with tf.GradientTape() as tape:
        #   y_c = self.input_model.get_layer(self.prop_from_layer).output[
        #           ..., self.cls] * self.roi.roi
        #   conv_output = self.input_model.get_layer(self.prop_to_layer).output

        # grads = tape.gradient(y_c,conv_output)

        # Normalize if necessary
        # grads = normalize(grads)
        tensor=tf.convert_to_tensor(input_details[0]['index'])
        gradient_function = K.function(tensor, [conv_output, grads])
        output, grads_val = gradient_function([preprocessed_input])
        self.A, self.grads_val = output[0, :], grads_val[0, :, :, :]

        return self.A, self.grads_val

    def gradientWeights(self):
        """Defines a matrix of alpha^k_c. Each alpha^k_c denotes importance (weights) of a feature map A^k for class c.
        If abs_w=True, absolute values of the matrix are processed and returned as weights.
        If posit_w=True, ReLU is applied to the matrix."""
        self.alpha_c = np.mean(self.grads_val, axis=(0, 1))
        if self.abs_w:
            self.alpha_c = abs(self.alpha_c)
        if self.posit_w:
            self.alpha_c = np.maximum(self.alpha_c, 0)

        return self.alpha_c

    def activationMap(self):
        """The last step to get the activation map. Should be called after outputGradients and gradientWeights."""
        # weighted sum of feature maps: sum of alpha^k_c * A^k
        cam = np.dot(self.A, self.alpha_c)  # *abs(grads_val) or max(grads_val,0)

        # Here it was modified to the same size as the original image 这里进行了修改，改成了输入图片（frame_proc）一致的尺寸
        cam = cv2.resize(cam, (self.image.shape[-3], self.image.shape[-2]), cv2.INTER_LINEAR)
        # apply ReLU to te sum
        cam = np.maximum(cam, 0)
        # normalize non-negative weighted sum
        self.cam_max = cam.max()
        if self.cam_max != 0 and self.normalize:
            cam = cam / self.cam_max
        self.cam = cam

        return self.cam

    def SGC(self):
        """Get the activation map"""
        _, _ = self.featureMapsGradients()
        _ = self.gradientWeights()

        return self.activationMap()

    def __sub__(self, otherSGC):
        """Subtraction experiment"""
        pass

    def average(self, otherSGCs):
        """average several seg-grad-cams"""
        new_sgc = self.copy()
        cam = self.SGC()
        cams = [cam]
        if otherSGCs is list:
            for other in otherSGCs:
                cams.append(other.SGC())
        else:
            cams.append(otherSGCs)

        aver = None
        for cc in cams:
            aver += cc
            print("aver shape: ", aver.shape)

        new_sgc.cam = aver / len(cams)
        return new_sgc

    def sortbyMax(self):
        """sort a list of seg-grad-cams by their maximum in activation map before normalization
        for f in sorted(listofSGCs, key = lambda x: x.sortbyMax()):
        print(f.image, f.cls, f.prop_to_layer, f.roi, f.cam_max)
        """
        return self.cam_max
