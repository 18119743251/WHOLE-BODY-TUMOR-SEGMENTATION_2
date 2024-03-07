import numpy as np


def non_zero_acc(pre, real):
    real = real.flatten()
    pre = pre.flatten()
    non_zero_real = real[((real == 1) | (pre == 1))]
    non_zero_pre = pre[((real == 1) | (pre == 1))]
    return np.sum(non_zero_real == non_zero_pre)/ len(non_zero_pre)

def np_dice(pre, real):
    k=1
    before_dice = np.sum(pre[real==k]==k)*2.0 / (np.sum(pre[pre==k]==k) + np.sum(real[real==k]==k))
    k=0
    back_dice = np.sum(pre[real==k]==k)*2.0 / (np.sum(pre[pre==k]==k) + np.sum(real[real==k]==k))
    return before_dice, back_dice

def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array, pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)

    false_pos = 0
    for idx in range(1, pred_conn_comp.max() + 1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask * gt_array).sum() == 0:
            false_pos = false_pos + comp_mask.sum()
    return false_pos


def false_neg_pix(gt_array, pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)

    false_neg = 0
    for idx in range(1, gt_conn_comp.max() + 1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask * pred_array).sum() == 0:
            false_neg = false_neg + comp_mask.sum()

    return false_neg
