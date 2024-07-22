import matplotlib.pyplot as plt
from gradcam.gradcam_tflite import SegGradCAM, ClassRoI
import numpy as np
import cv2


class SegGradCAMplot(SegGradCAM):
    def __init__(self, seggradcam, image, n_classes=None, outfolder=None, model=None):

        # SegGradCAM.__init__(self, seggradcam) #, trainparam) #?

        self.input_model = seggradcam.input_model  # 输入模型
        self.image = image  # 输入图片 raw image PIL
        self.image_pro = seggradcam.image  # the image after processing PIL
        self.cls = seggradcam.cls  # class 选择的类

        # prop_from_layer is the layer with logits prior to the last activation function 激活函数前的层
        self.prop_from_layer = seggradcam.prop_from_layer
        self.prop_to_layer = seggradcam.prop_to_layer  # an intermediate layer, typically of the bottleneck layers 中间层

        self.roi = seggradcam.roi  # M, a set of pixel indices of interest in the output mask. ROI
        self.normalize = seggradcam.normalize  # [True, False] normalize the saliency map L_c
        self.abs_w = seggradcam.abs_w  # if True, absolute function is applied to alpha_c
        self.posit_w = seggradcam.posit_w  # if True, ReLU is applied to alpha_c
        self.alpha_c = seggradcam.alpha_c  # alpha_c, weights for importance of feature maps
        self.A = seggradcam.A  # A, feature maps from the intermediate prop_to_layer
        self.grads_val = seggradcam.grads_val  # gradients of the logits y with respect to all pixels of each feature map 𝐴^𝑘
        self.cam = seggradcam.cam  # CAM image cv

        self.n_classes = n_classes  # 类的数量
        self.outfolder = outfolder  # 输出文件夹
        self.model = model  # 模型

        self.ximg = image  # used as the input for plotting
        self.cmap_orig = None

    def defaultScales(self):
        classes_cmap = plt.get_cmap('Spectral', self.n_classes)  # Colormaps
        scale_fig = 1  # 图片尺寸
        fonts = 70  # 字体
        scatter_size = 600 * scale_fig
        return classes_cmap, scale_fig, fonts, scatter_size

    def explainBase(self, title1, title1bias, start_save_name, pixel=True):
        """"""
        classes_cmap, scale_fig, fonts, scatter_size = self.defaultScales()
        fonts = int(fonts / 3)
        scatter_size = int(scatter_size / 3)
        plt.figure(figsize=(19.2, 10.8))
        # plt.axis('off')
        # self.ximg = np.squeeze(self.ximg)

        """
        Raw image plotting
        """
        plt.imshow(self.ximg, vmin=0, vmax=1, cmap=self.cmap_orig)
        # plt.show()

        """
        Segmentation outline plotting
        """
        # class contour
        # 画网格
        X, Y = self.roi.meshgrid()
     

        # (1, 256) -> (256)
        X = np.squeeze(X)
        Y = np.squeeze(Y)

        # reshape the shape of outline from (256, 256) to the original
        # shape of the input image e.g.(1920,1080)
        w, h = self.image.size
        X = X / 256 * w
        Y = Y / 256 * h

        if pixel:
            # i, j = self.roi.i, self.roi.j
            classroi = ClassRoI(self.model, self.image_pro, self.cls)
            roi_contour1 = classroi.roi
        else:
            roi_contour1 = self.roi.roi
        # 画等高线 roi_contour1是Z,第三维
        # plot Segmentation outline
        plt.contour(X, Y, roi_contour1, colors='pink')

        # """
        # Heatmap plotting
        # """
        plt.title(title1, fontsize=fonts)
        # biased texture contour
        # plot cam heatmap
        cam = self.cam
        # reshape the heatmap to the shape of the input image
        cam_resize = cv2.resize(cam, (self.image.size), interpolation=cv2.INTER_CUBIC)
        plt.imshow(cam_resize, cmap='jet',  # vmin=0,vmax=1,
                   alpha=0.6)
        jet = plt.colorbar(fraction=0.046, pad=0.04, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        jet.set_label(label="Importance", size=fonts)
        jet.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], size=fonts)

        plt.show()
        # plt.savefig(os.path.join(self.outfolder,
        #                          start_save_name + str(self.cls) + '_to_act' + self.prop_to_layer.split('_')[1] + '_' +
        #                          self.timestr + ".png"))

    def explainClass(self):
        """Plot seg-grad-cam explanation for a selected class channel"""
        title1 = 'Seg-Grad-CAM for Raindrops'
        title1bias = 'Seg-Grad-CAM for class %d \n& biased texture in magenta' % (self.cls)
        start_save_name = 'class'
        self.explainBase(title1, title1bias, start_save_name)

    def explainRoi(self):
        """Plot seg-grad-cam explanation for a region of interest"""
        title1 = 'Seg-Grad-CAM for RoI(pink) of class %d' % (self.cls)
        title1bias = 'Seg-Grad-CAM for RoI(pink) of class %d \n& biased texture in magenta' % (self.cls)
        start_save_name = 'roi_cl'
        self.explainBase(title1, title1bias, start_save_name)

    def explainPixel(self):
        """Plot seg-grad-cam explanation for a selected single pixel"""
        i, j = self.roi.i, self.roi.j
        title1 = 'Seg-Grad-CAM for pixel [%d,%d]. Class %d' % (i, j, self.cls)
        title1bias = 'Seg-Grad-CAM for pixel [%d,%d], class %d \n& biased texture in magenta' % (i, j, self.cls)
        start_save_name = 'pixel' + str(i) + '_' + str(j) + '_cl'
        self.explainBase(title1, title1bias, start_save_name, pixel=True)
