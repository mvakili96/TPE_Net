# 2020/7/12
# Jungwon Kang

import torch
import torch.nn as nn
import torch.nn.functional as F



########################################################################################################################
###
########################################################################################################################
def L1_loss(x_est, x_gt, b_sigmoid=False):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # x_est: (bs, 1, h, w), here 1 is the number of class
    # x_gt: (bs, 1, h, w)
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # L1 loss
    # https://pytorch.org/docs/master/generated/torch.nn.L1Loss.html

    if b_sigmoid is True:
        x_est = torch.clamp(x_est, min=1e-4, max=100)
    #end

    loss_a = nn.L1Loss()
    loss_b = loss_a(x_est, x_gt)

    return loss_b
#end


########################################################################################################################
###
########################################################################################################################
def MSE_loss(x_est, x_gt):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # x_est: (bs, 1, h, w), here 1 is the number of class
    # x_gt: (bs, 1, h, w)
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # MSELOSS
    # https://pytorch.org/docs/master/generated/torch.nn.MSELoss.html
    #
    # >> > loss = nn.MSELoss()
    # >> > input = torch.randn(3, 5, requires_grad=True)
    # >> > target = torch.randn(3, 5)
    # >> > output = loss(input, target)
    # >> > output.backward()

    #target = target.view(-1)


    x_est = torch.clamp(torch.sigmoid(x_est), min=1e-4, max=1 - 1e-4)

    #x_est = x_est.view(-1)
    #x_gt = x_gt.view(-1)

    loss_a = nn.MSELoss()
    loss_b = loss_a(x_est, x_gt)

    return loss_b
#end

# def regional_centerness_loss(cen_est, cen_gt, seg_gt)
#
#     for image_counter in range(cen_est.shape[0]):
#         for i in range(cen_est.shape[1]):
#             for j in range(cen_est.shape[2]):
#                 if seg_gt[image_counter,i,j] == 2:
#                     cen_est[image_counter,i,j] = cen_gt[image_counter,i,j]



########################################################################################################################
###
########################################################################################################################
def neg_loss(preds, targets):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # This function is from </home/yu1/PycharmProjects_b/proj_centernet_ut_austin/utils/losses.py>
    #
    # Modified focal loss. Exactly the same as CornerNet.
    #  Runs faster and costs a little bit more memory
    #  Arguments:
    #
    # preds: (bs, 1, h, w), here 1 is the number of class
    # targets: (bs, 1, h, w)
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////


    param_gamma = 4

    size_preds = preds.size()
    totnum_pixels = size_preds[0]*size_preds[1]*size_preds[2]*size_preds[3]


    pos_inds = targets.ge(0.5).float()          # pos_inds: (bs, 1, h, w)
    neg_inds = 1.0 - pos_inds                   # neg_inds: (bs, 1, h, w)
    #neg_inds = targets.lt(0.5).float()


    #pos_weights = torch.pow(targets, 4)         # pos_weights: (bs, 1, h, w)
        # giving more weight to targets:1
        #        less weight to targets:0

    #neg_weights = torch.pow(1 - targets, 4)     # neg_weights: (bs, 1, h, w)
        # giving more weight to targets:0
        #        less weight to targets:1

    # num_pos = pos_inds.float().sum()
    # num_neg = neg_inds.float().sum()
    # totnum = num_pos + num_neg
    # portion_pos = num_pos/totnum
    # portion_neg = num_neg/totnum
    # a = 1


    loss = 0

    for pred in preds:
        ###
        # pred: (1, h, w)
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
            # pred: (1, h, w)

        ###
        pos_loss = -1.0 * torch.log(pred) * torch.pow(1 - pred, param_gamma) * pos_inds
        neg_loss = -1.0 * torch.log(1 - pred) * torch.pow(pred, param_gamma) * neg_inds
            # pos_loss: (bs, 1, h, w)
            # neg_loss: (bs, 1, h, w)

        ###
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = loss + (pos_loss + neg_loss)
    #end

    return loss / totnum_pixels
#end


########################################################################################################################
###
########################################################################################################################
def neg_loss_cb(preds, targets):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # cb (class balanced version)
    # This function is from </home/yu1/PycharmProjects_b/proj_centernet_ut_austin/utils/losses.py>
    #
    # Modified focal loss. Exactly the same as CornerNet.
    #  Runs faster and costs a little bit more memory
    #  Arguments:
    #
    # preds:   (bs, 1, h, w), here 1 is the number of class
    # targets: (bs, 1, h, w)
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////


    param_gamma = 2

    size_preds = preds.size()
    #totnum_pixels = size_preds[0]*size_preds[1]*size_preds[2]*size_preds[3]


    ###
    pos_inds = targets.ge(0.5).float()          # pos_inds: (bs, 1, h, w)
    neg_inds = 1.0 - pos_inds                   # neg_inds: (bs, 1, h, w)


    ###
    num_pos = pos_inds.float().sum()
    num_neg = neg_inds.float().sum()
    totnum_all = num_pos + num_neg

    alpha_pos  = num_neg/totnum_all
    alpha_neg  = num_pos/totnum_all



    loss = 0

    for pred in preds:
        ###
        # pred: (1, h, w)
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
            # pred: (1, h, w)

        ###
        #pos_loss = -1.0 * torch.log(pred) * torch.pow(1 - pred, param_gamma) * pos_inds
        #neg_loss = -1.0 * torch.log(1 - pred) * torch.pow(pred, param_gamma) * neg_inds
        pos_loss = -1.0 * torch.log(pred) * pos_inds
        neg_loss = -1.0 * torch.log(1 - pred) * neg_inds
            # pos_loss: (bs, 1, h, w)
            # neg_loss: (bs, 1, h, w)

        ###
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = loss + (alpha_pos*pos_loss + alpha_neg*neg_loss)
    #end

    return loss/totnum_all
#end


########################################################################################################################
###
########################################################################################################################
def _neg_loss_ver0(preds, targets):

    ###============================================================================================================
    ### debugging
    ###============================================================================================================
    # print("preds")
    # print(type(preds))        # => tuple
    # print(len(preds))         # => Batch_size
    # print(preds[0].shape)       # => torch.Size([Batch_size, 80, 128, 128])
    # print(preds[0].min())       # Not 0.0. it can be (-) value.
    # print(preds[0].max())       # Not 1.0. it can be (-) value.


    # print("targets")
    # print(targets.type)
    # print(targets.shape)        # => torch.Size([Batch_size, 80, 128, 128]), where 80: total number of object classes
    # print(targets.min())        # usually, 0.0
    # print(targets.max())        # usually, 1.0


    # Note that
    #   len(preds) can become a value > 1, as a network can produce two output or more.
    #   However, targets is just one variable,
    #       as it can be used for any output of preds.


    ###============================================================================================================
    ###
    ###============================================================================================================

    ###
    pos_inds = targets == 1     # todo targets > 1-epsilon ?
    neg_inds = targets < 1      # todo targets < 1-epsilon ?

    # print("pos_inds")
    # print(pos_inds.shape)         # => torch.Size([Batch_size, 80, 128, 128]), where 80: total number of object classes
                                    #    where, each element in pos_inds is 0 or 1.

    # print("neg_inds")
    # print(neg_inds.shape)         # => torch.Size([Batch_size, 80, 128, 128]), where 80: total number of object classes
                                    #    where, each element in pos_inds is 0 or 1.


    ###
    neg_weights = torch.pow(1 - targets[neg_inds], 4)

    # print(targets[neg_inds].shape)      # => torch.Size([2621437])
    #                                     #    Here, 2621437 is one example, and it is exactly the same as size of neg_inds

    # print("neg_weights")
    # print(neg_weights.type)
    # print(neg_weights.shape)          # => torch.Size([2621437])
                                        #    Here, 2621437 is one example, and it is exactly the same as size of neg_inds


    ###============================================================================================================
    ###
    ###============================================================================================================

    ###
    loss = 0
    # idx = 0

    for pred in preds:
        # print("idx: [%d]" % idx)          # => index of output from one feedforwarding
        # idx += 1

        # print("pred")
        # print(pred.shape)       # => torch.Size([Batch_size, 80, 128, 128]), where 80: total number of object classes


        ###------------------------------------------------------------------------------------------------
        ### applying sigmoid, then clamp
        ###------------------------------------------------------------------------------------------------
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

        # print("pred (after clamp)")
        # print(pred.shape)       # => torch.Size([Batch_size, 80, 128, 128])


        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]


        # print("pos_pred")
        # print(pos_pred.shape)       # => torch.Size([3])
        #
        # print("neg_pred")
        # print(neg_pred.shape)       # => torch.Size([2621437])



        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights


        # print("pos_loss")
        # print(pos_loss.shape)       # => torch.Size([3])
        #
        # print("neg_loss")
        # print(neg_loss.shape)       # => torch.Size([2621437])



        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()


        # print("pos_loss")
        # print(pos_loss)             # => -3.8553
        # print(pos_loss.shape)       # => torch.Size([])
        #
        # print("neg_loss")
        # print(neg_loss)
        # print(neg_loss.shape)       # => torch.Size([])


        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        #end
    #end

    return loss / len(preds)
#end
########################################################################################################################


"""
########################################################################################################################
###
########################################################################################################################
def neg_loss(preds, targets):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # This function is from </home/yu1/PycharmProjects_b/proj_centernet_ut_austin/utils/losses.py>
    #
    # Modified focal loss. Exactly the same as CornerNet.
    #  Runs faster and costs a little bit more memory
    #  Arguments:
    #  preds (B x c x h x w)
    #  gt_regr (B x c x h x w)
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////

    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)

    loss = 0

    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        #end
    #end

    return loss / len(preds)
#end
"""



"""
########################################################################################################################
###
########################################################################################################################
def neg_loss(preds, targets):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # This function is from </home/yu1/PycharmProjects_b/proj_centernet_ut_austin/utils/losses.py>
    #
    # Modified focal loss. Exactly the same as CornerNet.
    #  Runs faster and costs a little bit more memory
    #  Arguments:
    #  preds (B x c x h x w)
    #  gt_regr (B x c x h x w)
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # preds: (bs, 1, h, w), here 1 is the number of class
    # targets: (bs, 1, h, w)

    pos_inds = targets.eq(1).float()            # pos_inds: (bs, 1, h, w)
    neg_inds = targets.lt(1).float()            # neg_inds: (bs, 1, h, w)

    neg_weights = torch.pow(1 - targets, 4)     # neg_weights: (bs, 1, h, w)
        # giving more weight to targets:0
        #        less weight to targets:1


    loss = 0

    for pred in preds:
        # pred: (1, h, w)
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
            # pred: (1, h, w)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
            # pos_loss: (bs, 1, h, w)
            # neg_loss: (bs, 1, h, w)


        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        #end
    #end

    return loss / len(preds)
#end
"""



"""
########################################################################################################################
###
########################################################################################################################
def _reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)
#end
"""


"""
########################################################################################################################
###
########################################################################################################################
def neg_loss(preds, targets):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # This function is from </home/yu1/PycharmProjects_b/proj_centernet_ut_austin/utils/losses.py>
    #
    # Modified focal loss. Exactly the same as CornerNet.
    #  Runs faster and costs a little bit more memory
    #  Arguments:
    #  preds (B x c x h x w)
    #  gt_regr (B x c x h x w)
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # preds: (bs, 1, h, w), here 1 is the number of class
    # targets: (bs, 1, h, w)

    pos_inds = targets.eq(1).float()            # pos_inds: (bs, 1, h, w)
    neg_inds = targets.lt(1).float()            # neg_inds: (bs, 1, h, w)

    neg_weights = torch.pow(1 - targets, 4)     # neg_weights: (bs, 1, h, w)
        # giving more weight to targets:0
        #        less weight to targets:1


    loss = 0

    for pred in preds:
        # pred: (1, h, w)
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
            # pred: (1, h, w)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
            # pos_loss: (bs, 1, h, w)
            # neg_loss: (bs, 1, h, w)


        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        #end
    #end

    return loss / len(preds)
#end
"""



