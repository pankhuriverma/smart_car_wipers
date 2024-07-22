import numpy as np


# calculate the dice coefficient
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001

    res = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    return res

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    if union.any() == 0:
        res = 1
    else:
        res = np.sum(intersection) / np.sum(union)
    return res


def pxl_acc(y_true, y_pred):
    correct_pxls = np.sum(y_true == y_pred)
    total_pxls = y_true.size
    res = correct_pxls / total_pxls

    return res


'''def pxl_acc(y_true, y_pred):
    correct_pxls = np.sum(np.logical_and(y_true, y_pred))
    total_pxls = y_true.size
    res = correct_pxls / total_pxls

    return res'''
