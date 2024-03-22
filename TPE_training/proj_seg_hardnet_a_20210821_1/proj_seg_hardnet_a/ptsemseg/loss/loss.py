import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.autograd import Variable as V
import keras.backend as K

########################################################################################################################
###
########################################################################################################################
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    
    loss = F.cross_entropy(
              input, target, weight=weight, size_average=size_average, ignore_index=250, reduction='mean')

    return loss



########################################################################################################################
###
########################################################################################################################
def multi_scale_cross_entropy2d(input, target, loss_th, weight=None, size_average=True, scale_weight=[1.0, 0.4]):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    K = input[0].size()[2] * input[0].size()[3] // 128
    loss = 0.0

    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * bootstrapped_cross_entropy2d(
            input=inp, target=target, min_K=K, loss_th=loss_th, weight=weight, size_average=size_average
        )

    return loss


########################################################################################################################
###
########################################################################################################################
def bootstrapped_cross_entropy2d(input, target, min_K, loss_th, weight=None, size_average=True, train_val = 0, dev = None):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    batch_size = input.size()[0]
    
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    
    thresh = loss_th
    
    def _bootstrap_xentropy_single(input, target, K, thresh, weight=None, size_average=True, train_validation = 0):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        # for i,item in enumerate(target):
        #     if item == 1:
        #         indices.append(i)
        if train_validation == 0:
            sorted_loss, _ = torch.sort(loss, descending=True)

            if sorted_loss[K] > thresh:
                loss = sorted_loss[sorted_loss > thresh]
            else:
                loss = sorted_loss[:K]

        reduced_topk_loss = torch.mean(loss)

        return reduced_topk_loss
    #end

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=min_K,
            thresh=thresh,
            weight=weight,
            size_average=size_average,
            train_validation = train_val,
        )
    # print(loss / float(batch_size))
    return loss / float(batch_size)


def dice_ce_loss(input, target, min_K, loss_th, weight=None, size_average=True, train_val = 0, dev = None ):
    ce_loss = cross_entropy2d(input, target)
    eps=1e-7
    true_1_hot = torch.eye(3)[target.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(input, dim=1)
    true_1_hot = true_1_hot.type(input.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss) + ce_loss


def CrossEntropyLoss2d_Weighted(input, target, min_K, loss_th, weight=None, size_average=True, train_val = 0, dev = None ):
    class_weights = torch.ones(3)
    class_weights[0] = 2
    class_weights[1] = 10
    class_weights[2] = 0.1
    class_weights = class_weights.to(dev)
    NLLloss = torch.nn.NLLLoss2d(class_weights)
    return NLLloss(torch.nn.functional.log_softmax(input, dim=1), target)


def PolyLaneNet_loss(outputs,
         target,
         conf_weight=1,
         lower_weight=1,
         upper_weight=1,
         cls_weight=1,
         poly_weight=300,
         threshold=15 / 720.):
    pred = outputs
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    s = nn.Sigmoid()
    threshold = nn.Threshold(threshold**2, 0.)
    pred = pred.reshape(-1, target.shape[1], 9)

    target_categories, pred_confs     = target[:, :, 0].reshape((-1, 1)), s(pred[:, :, 0]).reshape((-1, 1))
    target_uppers_h, pred_uppers_h    = target[:, :, 2].reshape((-1, 1)), pred[:, :, 2].reshape((-1, 1))
    target_uppers_w, pred_uppers_w    = target[:, :, 4].reshape((-1, 1)), pred[:, :, 4].reshape((-1, 1))
    target_points, pred_polys         = target[:, :, 5:].reshape((-1, target.shape[2] - 5)), pred[:, :, 5:].reshape(-1, 4)
    target_lowers_h, pred_lowers_h    = target[:, :, 1].reshape((-1, 1)), pred[:, :, 1].reshape((-1, 1))
    target_lowers_w, pred_lowers_w    = target[:, :, 3].reshape((-1, 1)), pred[:, :, 3].reshape((-1, 1))


    target_confs = (target_categories > 0).float()

    valid_lanes_idx = target_confs == 1
    valid_lanes_idx_flat = valid_lanes_idx.reshape(-1)




    # batch_size = target.shape[0]
    # for i in range(0,batch_size):
    #     PRED_vectors_cur_img = pred[i]
    #     GT_vectors_cur_img   = target[i]
    #
    #     GT_categories, PRED_confs = GT_vectors_cur_img[:, 0].reshape((-1, 1)), s(PRED_vectors_cur_img[:, 0]).reshape((-1, 1))
    #
    #     valid_GT_lanes_idx = GT_categories == 1
    #     valid_GT_lanes_idx_flat = valid_GT_lanes_idx.reshape(-1)
    #
    #     valid_PRED_lanes_idx = PRED_confs >= 0.5
    #     valid_PRED_lanes_idx_flat = valid_PRED_lanes_idx.reshape(-1)
    #
    #     GT_uppers, PRED_uppers    = GT_vectors_cur_img[valid_GT_lanes_idx_flat, 2].reshape((-1, 1)), PRED_vectors_cur_img[valid_PRED_lanes_idx_flat, 2].reshape((-1, 1))





    lower_loss_h = mse(target_lowers_h[valid_lanes_idx], pred_lowers_h[valid_lanes_idx])
    upper_loss_h = mse(target_uppers_h[valid_lanes_idx], pred_uppers_h[valid_lanes_idx])
    lower_loss_w = mse(target_lowers_w[valid_lanes_idx], pred_lowers_w[valid_lanes_idx])
    upper_loss_w = mse(target_uppers_w[valid_lanes_idx], pred_uppers_w[valid_lanes_idx])


    # poly loss calc
    target_xs = target_points[valid_lanes_idx_flat, :target_points.shape[1] // 2]
    ys        = target_points[valid_lanes_idx_flat,  target_points.shape[1] // 2:].t()
    valid_xs = target_xs >= -1
    pred_polys = pred_polys[valid_lanes_idx_flat]
    pred_xs = pred_polys[:, 0] * ys**3 + pred_polys[:, 1] * ys**2 + pred_polys[:, 2] * ys + pred_polys[:, 3]
    pred_xs.t_()
    weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5
    pred_xs = (pred_xs.t_() * weights).t()
    target_xs = (target_xs.t_() * weights).t()
    # poly_loss = mse(pred_xs[valid_xs], target_xs[valid_xs]) / valid_lanes_idx.sum()
    poly_loss = threshold(
       (pred_xs[valid_xs] - target_xs[valid_xs])**2).sum() / (valid_lanes_idx.sum() * valid_xs.sum())

    # applying weights to partial losses
    poly_loss  = poly_loss * poly_weight
    lower_loss_h = lower_loss_h * lower_weight
    upper_loss_h = upper_loss_h * upper_weight
    conf_loss  = bce(pred_confs, target_confs) * conf_weight

    print(poly_loss)
    print(lower_loss_h)
    print(upper_loss_h)
    print(conf_loss)

    loss = upper_loss_h + lower_loss_h + conf_loss + poly_loss + upper_loss_w + lower_loss_w


    # print(loss)
    # print(target_lowers[valid_lanes_idx])
    # print(pred_lowers[valid_lanes_idx])

    return loss

    # # {
    #     'conf': conf_loss,
    #     'lower': lower_loss,
    #     'upper': upper_loss,
    #     'poly': poly_loss,
    #     'cls_loss': cls_loss
    #     }