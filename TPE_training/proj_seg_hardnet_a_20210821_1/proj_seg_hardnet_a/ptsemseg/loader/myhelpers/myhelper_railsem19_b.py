# 2020/7/10
# Jungwon Kang


import os
import torch
#import torch.nn as nn
import numpy as np
import collections
import cv2
import math
import copy
from scipy.signal import find_peaks
import tifffile as tiff
from torchvision.transforms import ToTensor
from ptsemseg.loader.myhelpers.lib.image import draw_umich_gaussian, gaussian_radius



########################################################################################################################
###
########################################################################################################################
def read_img_raw_jpg_from_file(full_fname_img_raw_jpg, size_img_rsz,arch,
                                rgb_mean = np.array([128.0, 128.0, 128.0]) / 255.0,
                                rgb_std = np.array([1.0, 1.0, 1.0])):


    #===================================================================================================
    # read img_raw_jpg from file
    # <input>
    #  full_fname_img_raw_jpg: ndarray(H,W,C), 0~255
    # <output>
    #  img_raw_rsz_fl_n: ndarray(C,H,W), -1.0 ~ 1.0, BGR
    #===================================================================================================

    ###================================================================================================
    ### read img_raw_jpg
    ###================================================================================================
    # print(full_fname_img_raw_jpg)
    img_raw = cv2.imread(full_fname_img_raw_jpg)
        # completed to set
        #       img_raw: ndarray(H,W,C), 0 ~ 255

        # Note that opencv uses BGR, that is:
        #   img_raw[:,:,0] -> B
        #   img_raw[:,:,1] -> G
        #   img_raw[:,:,2] -> R


    ###================================================================================================
    ### resize img
    ###================================================================================================
    img_raw_rsz_uint8 = cv2.resize(img_raw, (size_img_rsz['w'], size_img_rsz['h']))
        # completed to set
        #       img_raw_rsz_uint8

    ### <<debugging>>
    if 0:
        cv2.imshow('img_raw_rsz_uint8', img_raw_rsz_uint8)
        cv2.waitKey(1)
    # end


    ###================================================================================================
    ### convert img_raw to img_data
    ###================================================================================================
    img_raw_rsz_fl_n_final = convert_img_ori_to_img_data(img_raw_rsz_uint8, arch)
        # completed to set
        #       img_raw_rsz_fl_n_final: ndarray(C,H,W), -X.0 ~ X.0


    ### <<debugging>>
    if 0:
        img_raw_temp0 = convert_img_data_to_img_ori(img_raw_rsz_fl_n_final)
        cv2.imshow('img_raw_temp0', img_raw_temp0)
        cv2.waitKey(1)
    # end


    return img_raw_rsz_uint8, img_raw_rsz_fl_n_final
#end


########################################################################################################################
###
########################################################################################################################
def convert_img_ori_to_img_data(img_ori_uint8,
                                arch,
                                rgb_mean=np.array([128.0, 128.0, 128.0]) / 255.0,
                                rgb_std=np.array([1.0, 1.0, 1.0])):

    #/////////////////////////////////////////////////////////////////////////////////////////////////////////
    # convert img_ori to img_data
    # <input>
    #   img_ori_uint8:      ndarray(H,W,C), 0 ~ 255
    # <output>
    #   img_raw_fl_n_final: ndarray(C,H,W), -X.0 ~ X.0
    #
    # we are doing the following things:
    #   (1) normalize so that 0~255 -> 0.0~1.0
    #   (2) apply rgb_mean
    #   (3) apply rgb_std
    #   (4) convert HWC -> CHW
    #   (5) make sure it is float32 type
    #/////////////////////////////////////////////////////////////////////////////////////////////////////////

    if arch == "rpnet_c" or arch == "bisenet_v2":
        img_ori_fl = img_ori_uint8.astype(np.float32) / 255.0
        img_ori_fl_n = img_ori_fl - rgb_mean
        img_ori_fl_n = img_ori_fl_n / rgb_std
        img_ori_fl_n = img_ori_fl_n.transpose(2, 0, 1)
        img_data_fl_n_final = img_ori_fl_n.astype(np.float32)
    elif arch == "res_101":
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        IMAGENET_STD = np.array([0.229, 0.224, 0.225])
        img_ori_fl = img_ori_uint8.astype(np.float32) / 255.0
        img_ori_fl_n = img_ori_fl - IMAGENET_MEAN
        img_ori_fl_n = img_ori_fl_n / IMAGENET_STD
        img_ori_fl_n = img_ori_fl_n.transpose(2, 0, 1)
        img_data_fl_n_final = img_ori_fl_n.astype(np.float32)
    elif arch == "dlinknet_34":
        img_data_fl_n_final = np.array(img_ori_uint8, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    elif arch == "erfnet":
        img_data_fl_n_final = ToTensor()(img_ori_uint8)
        img_data_fl_n_final = img_data_fl_n_final.numpy()
    else:
        raise ("No model found for converting image to data")


    return img_data_fl_n_final
        # ndarray(C,H,W), -X.0 ~ X.0
#end


########################################################################################################################
###
########################################################################################################################
def convert_img_data_to_img_ori(img_data_fl_n,
                                rgb_mean=np.array([128.0, 128.0, 128.0]) / 255.0,
                                rgb_std=np.array([1.0, 1.0, 1.0])):

    #/////////////////////////////////////////////////////////////////////////////////////////////////////////
    # convert img_data to img_raw
    # <output>
    #   img_data_fl_n: ndarray(C,H,W), -X.0 ~ X.0, BGR
    # <input>
    #   img_ori_uint8: ndarray(H,W,C), 0 ~ 255, BGR
    #
    #   we are doing the following things:
    #   (1) convert CHW -> HWC
    #   (2) apply rgb_std
    #   (3) apply rgb_mean
    #   (4) de-normalize so that 0.0~1.0 -> 0~255
    #   (5) make it uint8
    #/////////////////////////////////////////////////////////////////////////////////////////////////////////

    img_out = copy.deepcopy(img_data_fl_n)

    ###================================================================================================
    ### (1) convert CHW -> HWC
    ###================================================================================================
    img_data_fl_n = img_data_fl_n.transpose(1, 2, 0)
        # C(0),H(1),W(2) -> H(1),W(2),C(0)
        # completed to set
        #       img_data_fl_n: ndarray(H,W,C)


    ###================================================================================================
    ### (2) apply rgb_std
    ###================================================================================================
    img_data_fl_n = img_data_fl_n*rgb_std


    ###================================================================================================
    ### (3) apply rgb_mean
    ###================================================================================================
    img_data_fl_n = img_data_fl_n + rgb_mean


    ###================================================================================================
    ### (4) de-normalize so that 0.0~1.0 -> 0~255
    ###================================================================================================
    img_data_fl_n = img_data_fl_n*255.0


    ###================================================================================================
    ### (5) make it uint8
    ###================================================================================================
    img_ori_uint8 = img_data_fl_n.astype(np.uint8)


    return img_ori_uint8
#end


########################################################################################################################
###
########################################################################################################################
def read_label_seg_png_from_file(full_fname_label_seg_png, size_img_rsz):

    ###================================================================================================
    ### read label_seg_png
    ###================================================================================================
    img_raw = cv2.imread(full_fname_label_seg_png, cv2.IMREAD_GRAYSCALE)
        # completed to set
        #       img_raw: ndarray(H,W,C), 0 ~ 255


    ###================================================================================================
    ### resize img
    ###================================================================================================
    img_raw_rsz_uint8 = cv2.resize(img_raw, (size_img_rsz['w'], size_img_rsz['h']))
        # completed to set
        #       img_raw_rsz_uint8

    return img_raw_rsz_uint8
#end

########################################################################################################################
###
########################################################################################################################
def read_triplet_image_from_file(full_fname_triplet_image_png,size_img_rsz):

    ###================================================================================================
    ### read label_seg_png
    ###================================================================================================
    img_raw = cv2.imread(full_fname_triplet_image_png,cv2.IMREAD_GRAYSCALE)
        # completed to set
        #       img_raw: ndarray(H,W,C), 0 ~ 255
    if size_img_rsz['w'] != 960 or size_img_rsz['h'] != 540:
        img_raw_rsz = cv2.resize(img_raw, (size_img_rsz['w'], size_img_rsz['h']))
    else:
        img_raw_rsz = img_raw


    img_reshaped = np.array([img_raw_rsz])

    return img_reshaped
#end

########################################################################################################################
###
########################################################################################################################
def read_triplet_C_from_file(full_fname_triplet_C_tiff,size_img_rsz):

    ###================================================================================================
    ### read label_seg_png
    ###================================================================================================
    img_raw = tiff.imread(full_fname_triplet_C_tiff)
        # completed to set
        #       img_raw: ndarray(H,W,C), 0 ~ 255
    if size_img_rsz['w'] != 960 or size_img_rsz['h'] != 540:
        img_raw_rsz = cv2.resize(img_raw, (size_img_rsz['w'], size_img_rsz['h']))
    else:
        img_raw_rsz = img_raw

    img_reshaped = np.array([img_raw_rsz])
    return img_reshaped
#end

########################################################################################################################
###
########################################################################################################################
def read_fnames_trainval(dir_this, idx_split):

    ### read fname for all the files at dir_this
    list_fname_ = os.listdir(dir_this)
    list_fname  = sorted(list_fname_)
        # completed to set
        #       list_fname: list for fnames only, (e.g. rs00001.jpg)


    ### store fnames according to train/val
    list_fname_train = list_fname[0:idx_split]
    list_fname_val   = list_fname[idx_split:]


    ### store
    dict_fnames = collections.defaultdict(list)
    dict_fnames["train"] = list_fname_train
    dict_fnames["val"]   = list_fname_val
        # completed to set
        #       dict_fnames["train"]: list for fnames only (for train)
        #       dict_fnames["val"]  : list for fnames only (for val)
        #               (e.g. 'rs01001.jpg', 'rs01002.jpg', ...)


    return dict_fnames
#end


########################################################################################################################
###
########################################################################################################################
def create_hmap_centerline(list_triplet_json, size_hmap, FACTOR_ori_to_hmap_h, FACTOR_ori_to_hmap_w):
    #===================================================================================================
    # list_triplet_set: each list element is {list: N},
    #                   where, each list sub-element in the list element is {list:4} (x_L, x_C, x_R, y)
    #
    #   Note that (y,x) in list_triplet_set are wrt original img size
    #===================================================================================================

    ###================================================================================================
    ###
    ###================================================================================================
    num_classes = 1

    h_hmap = size_hmap['h']
    w_hmap = size_hmap['w']


    ###
    hmap_out = np.zeros((num_classes, size_hmap['h'], size_hmap['w']), dtype=np.float32)
    hmap_out0 = hmap_out[0]


    ###================================================================================================
    ###
    ###================================================================================================
    mode_vote = 1

    for list_this_set in list_triplet_json:
        # list_this_set: {list: N}, consists of N triplets.

        for triplet_this in list_this_set:
            # triplet_this: {list:4} (x_L, x_C, x_R, y)


            ### get this triplet
            x_L_fl  = triplet_this[0]*FACTOR_ori_to_hmap_w
            x_C_fl  = triplet_this[1]*FACTOR_ori_to_hmap_w
            x_R_fl  = triplet_this[2]*FACTOR_ori_to_hmap_w
            y       = int(round(triplet_this[3]*FACTOR_ori_to_hmap_h))

            if (y < 0) or (y >= h_hmap):
                continue
            #end

            ###
            x_L_int = round(x_L_fl)
            x_R_int = round(x_R_fl)


            ###
            if (x_L_int < 0) or (x_R_int < 0) or (x_L_int >= w_hmap) or (x_R_int >= w_hmap):
                continue
            # end


            if mode_vote == 0:
                hmap_out0[y, x_L_int:(x_R_int + 1)] = 1.0

            elif mode_vote == 1:
                ###
                half_w_triplet_ = (x_R_fl - x_L_fl)*0.5
                half_w_triplet  = max(1.0, half_w_triplet_)

                for x_this in range(x_L_int, x_R_int + 1):
                    ###
                    if (x_this < 0) or (x_this >= w_hmap):
                        continue
                    #end

                    ###
                    dx = abs(x_this - x_C_fl)
                    alpha = dx/half_w_triplet
                    val_heat_ = 1.0 - alpha
                    val_heat  = max(0.0, val_heat_)

                    val_heat_old = hmap_out0[y, x_this]

                    hmap_out0[y, x_this] = max(val_heat, val_heat_old)
                #end
            #end
        #end
    #end
        # completed to set
        #       hmap_out


    return hmap_out
#end


########################################################################################################################
###
########################################################################################################################

def create_labelmap_triplet_priority(list_triplet_json, size_labelmap, FACTOR_ori_to_labelmap_h, FACTOR_ori_to_labelmap_w):
    num_classes = 1

    h_labelmap = size_labelmap['h']
    w_labelmap = size_labelmap['w']

    labelmap_out = np.zeros((num_classes, size_labelmap['h'], size_labelmap['w']), dtype=np.float32)
    labelmap_out0 = labelmap_out[0]

    for list_this_set in list_triplet_json:
        # list_this_set: {list: N}, consists of N triplets.

        for triplet_this in list_this_set:
            # triplet_this: {list:4} (x_L, x_C, x_R, y)

            ###---------------------------------------------------------------------------------------
            ### get this triplet
            ###---------------------------------------------------------------------------------------
            x_L_fl  = triplet_this[0]*FACTOR_ori_to_labelmap_w
            x_C_fl  = triplet_this[1]*FACTOR_ori_to_labelmap_w
            x_R_fl  = triplet_this[2]*FACTOR_ori_to_labelmap_w
            y       = int(round(triplet_this[3]*FACTOR_ori_to_labelmap_h))

            if (y < 0) or (y >= h_labelmap):
                continue
            #end

            ###
            x_L_int = round(x_L_fl)
            x_R_int = round(x_R_fl)


            ###
            if (x_L_int < 0) or (x_R_int < 0) or (x_L_int >= w_labelmap) or (x_R_int >= w_labelmap):
                continue
            # end


            ###---------------------------------------------------------------------------------------
            ### vote for centerline
            ###---------------------------------------------------------------------------------------
            for x_this in range(x_L_int, x_R_int + 1):
                if (x_this < 0) or (x_this >= w_labelmap):
                    continue

                if abs(x_this - round(x_C_fl)) <= 5:
                    dist_average = 1
                else:
                    dist_average = 0

                val_heat_old = labelmap_out0[y, x_this]
                labelmap_out0[y, x_this] = max(dist_average,val_heat_old)
    # print(labelmap_out.shape)
    # cv2.imshow('A',labelmap_out[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return labelmap_out


########################################################################################################################
###
########################################################################################################################

def create_labelmap_triplet(list_triplet_json, size_labelmap, FACTOR_ori_to_labelmap_h, FACTOR_ori_to_labelmap_w):
    #===================================================================================================
    # list_triplet_set: each list element is {list: N},
    #                   where, each list sub-element in the list element is {list:4} (x_L, x_C, x_R, y)
    #
    #   Note that (y,x) in list_triplet_set are wrt original img size
    #===================================================================================================

    ###================================================================================================
    ###
    ###================================================================================================
    num_classes = 1

    h_labelmap = size_labelmap['h']
    w_labelmap = size_labelmap['w']
    # print(h_labelmap)


    ###
    labelmap_centerline_out = np.zeros((num_classes, size_labelmap['h'], size_labelmap['w']), dtype=np.float32)
    labelmap_leftrail_out   = np.zeros((num_classes, size_labelmap['h'], size_labelmap['w']), dtype=np.float32)
    labelmap_rightrail_out  = np.zeros((num_classes, size_labelmap['h'], size_labelmap['w']), dtype=np.float32)


    labelmap_centerline_out0 = labelmap_centerline_out[0]
    labelmap_leftrail_out0   = labelmap_leftrail_out[0]
    labelmap_rightrail_out0  = labelmap_rightrail_out[0]


    ###================================================================================================
    ###
    ###================================================================================================
    mode_vote = 1

    for list_this_set in list_triplet_json:
        # list_this_set: {list: N}, consists of N triplets.

        for triplet_this in list_this_set:
            # triplet_this: {list:4} (x_L, x_C, x_R, y)

            ###---------------------------------------------------------------------------------------
            ### get this triplet
            ###---------------------------------------------------------------------------------------
            x_L_fl  = triplet_this[0]*FACTOR_ori_to_labelmap_w
            x_C_fl  = triplet_this[1]*FACTOR_ori_to_labelmap_w
            x_R_fl  = triplet_this[2]*FACTOR_ori_to_labelmap_w
            y       = int(round(triplet_this[3]*FACTOR_ori_to_labelmap_h))

            if (y < 0) or (y >= h_labelmap):
                continue
            #end

            ###
            x_L_int = round(x_L_fl)
            x_R_int = round(x_R_fl)


            ###
            if (x_L_int < 0) or (x_R_int < 0) or (x_L_int >= w_labelmap) or (x_R_int >= w_labelmap):
                continue
            # end


            ###---------------------------------------------------------------------------------------
            ### vote for centerline
            ###---------------------------------------------------------------------------------------
            if mode_vote == 0:
                labelmap_centerline_out0[y, x_L_int:(x_R_int + 1)] = 1.0

            elif mode_vote == 1:
                for x_this in range(x_L_int, x_R_int + 1):
                    if (x_this < 0) or (x_this >= w_labelmap):
                        continue
                    dist_this_to_L = abs(x_this - x_L_fl)
                    dist_this_to_R = abs(x_R_fl - x_this)
                    dist_average = min(dist_this_to_L,dist_this_to_R)

                    val_heat_old = labelmap_centerline_out0[y, x_this]
                    labelmap_centerline_out0[y, x_this] = max(dist_average,val_heat_old)


                # half_w_triplet_ = (x_R_fl - x_L_fl)*0.5
                # half_w_triplet  = max(1.0, half_w_triplet_)
                #
                # for x_this in range(x_L_int, x_R_int + 1):
                #     ###
                #     if (x_this < 0) or (x_this >= w_labelmap):
                #         continue
                #     #end
                #
                #     ###
                #     dx = abs(x_this - x_C_fl)
                #     alpha = (dx/half_w_triplet)
                #
                #     val_heat_ = (1.0 - alpha)
                #     val_heat  = max(0.0, val_heat_)
                #
                #     val_heat_old = labelmap_centerline_out0[y, x_this]
                #
                #     labelmap_centerline_out0[y, x_this] = max(val_heat, val_heat_old)

                #end
            #end
                # completed to set (for one triplet)
                #       labelmap_centerline_out0


            ###---------------------------------------------------------------------------------------
            ### vote for left & right rails
            ###---------------------------------------------------------------------------------------
            x_fill_s = x_L_int
            x_fill_e = x_R_int

            for x_fill_this in range(x_fill_s, x_fill_e + 1):
                ###
                if (x_fill_this < 0) or (x_fill_this >= w_labelmap):
                    continue
                # end

                dist_this_to_L = abs(x_fill_this - x_L_fl)
                dist_this_to_R = abs(x_R_fl - x_fill_this)

                labelmap_leftrail_out0[y, x_fill_this] = dist_this_to_L
                labelmap_rightrail_out0[y, x_fill_this] = dist_this_to_R
            #end
                # completed to set  (for one triplet)
                #       labelmap_leftrail_out0
                #       labelmap_rightrail_out0
        #end
    #end
        # completed to set
        #       labelmap_centerline_out
        #       labelmap_leftrail_out
        #       labelmap_rightrail_out

    return labelmap_centerline_out, labelmap_leftrail_out, labelmap_rightrail_out
#end




# for x_this in range(x_L, x_R + 1):
# end
# pnt_center = [x_C, y]
# draw_umich_gaussian(hmap_out[0], pnt_center, radius)

### compute params for vote
# width_rail_ = x_R - x_L
# width_rail  = max(param_width_min, width_rail_)*1.5
# radius = max(5, int(gaussian_radius((math.ceil(width_rail), math.ceil(width_rail)), gaussian_iou)))
# completed to set
#       radius


########################################################################################################################
###
########################################################################################################################
def visualize_label_my_triplet(list_triplet_set, img_bg, FACTOR_ori_to_rsz_h, FACTOR_ori_to_rsz_w):
    #===================================================================================================
    # visualize label my triplet
    # <input>
    # list_triplet_set: each list element is {list: N},
    #                   where, each list sub-element in the list element is {list:4} (x_L, x_C, x_R, y)
    # img_bg: deep-copied image
    #===================================================================================================

    ### <For visualization>
    bgr_my_triplet = {"background": (0, 0, 0),
                      "rail_left":  (0, 0, 255),
                      "centerline": (0, 255, 0),
                      "rail_right": (255, 0, 0),
                      }

    ###
    for list_this_set in list_triplet_set:
        # list_this_set: {list: N}, consists of N triplets.

        for triplet_this in list_this_set:
            # triplet_this: {list:4} (x_L, x_C, x_R, y)

            ###
            x_L = int(round(triplet_this[0]*FACTOR_ori_to_rsz_w))
            x_C = int(round(triplet_this[1]*FACTOR_ori_to_rsz_w))
            x_R = int(round(triplet_this[2]*FACTOR_ori_to_rsz_w))
            y   = int(round(triplet_this[3]*FACTOR_ori_to_rsz_h))

            ###
            img_bg[y, x_L] = bgr_my_triplet["rail_left"]
            img_bg[y, x_C] = bgr_my_triplet["centerline"]
            img_bg[y, x_R] = bgr_my_triplet["rail_right"]
        # end
    # end



    ###
    if 1:
        cv2.imshow('img_vis_label_my_triplet', img_bg)
        cv2.waitKey()
    #end

    return
#end


########################################################################################################################
###
########################################################################################################################
def visualize_labelmaps(labelmap_center, labelmap_left, labelmap_right, img_raw_in):
    #===============================================================================================================
    # visualize hmap
    # <input>
    #  hmap_this: ndarr(h, w), 0.0 ~ 1.0
    #===============================================================================================================


    ############################################################################################################
    ### 0. centerline
    ############################################################################################################

    ###
    img_labelmap_center_ = labelmap_center*255.0
    img_labelmap_center  = img_labelmap_center_.astype(np.uint8)
    img_labelmap_center_rgb = cv2.cvtColor(img_labelmap_center, cv2.COLOR_GRAY2BGR)

    ###
    img_labelmap_center_final = cv2.addWeighted(src1=img_raw_in, alpha=0.25, src2=img_labelmap_center_rgb, beta=0.75, gamma=0)

    ###
    if 0:
        cv2.imshow('img_labelmap_center', img_labelmap_center_final)
        cv2.waitKey(1)
    #end


    ############################################################################################################
    ### 1. left rail
    ############################################################################################################

    ###
    img_labelmap_left_ = labelmap_left
    img_labelmap_left  = img_labelmap_left_.astype(np.uint8)
    img_labelmap_left_rgb = cv2.cvtColor(img_labelmap_left, cv2.COLOR_GRAY2BGR)

    ###
    img_labelmap_left_final = cv2.addWeighted(src1=img_raw_in, alpha=0.25, src2=img_labelmap_left_rgb, beta=0.75, gamma=0)

    ###
    if 0:
        cv2.imshow('img_labelmap_left', img_labelmap_left_rgb_final)
        cv2.waitKey(1)
    #end


    ############################################################################################################
    ### 2. right rail
    ############################################################################################################

    ###
    img_labelmap_right_ = labelmap_right
    img_labelmap_right  = img_labelmap_right_.astype(np.uint8)
    img_labelmap_right_rgb = cv2.cvtColor(img_labelmap_right, cv2.COLOR_GRAY2BGR)

    ###
    img_labelmap_right_final = cv2.addWeighted(src1=img_raw_in, alpha=0.25, src2=img_labelmap_right_rgb, beta=0.75, gamma=0)

    ###
    if 0:
        cv2.imshow('img_labelmap_right_final', img_labelmap_right_final)
        cv2.waitKey(1)
    #end


    return img_labelmap_center_rgb, img_labelmap_left_rgb, img_labelmap_right_rgb
#end


########################################################################################################################
###
########################################################################################################################
def visualize_hmap(hmap_this, img_raw_in):
    #===================================================================================================
    # visualize hmap
    # <input>
    #  hmap_this: ndarr(h, w), 0.0 ~ 1.0
    #===================================================================================================

    ###
    img_hmap_ = hmap_this*255.0
    img_hmap  = img_hmap_.astype(np.uint8)
    img_hmap_rgb = cv2.cvtColor(img_hmap, cv2.COLOR_GRAY2BGR)


    ###
    img_hmap_final = cv2.addWeighted(src1=img_raw_in, alpha=0.25, src2=img_hmap_rgb, beta=0.75, gamma=0)
    cv2.imshow('img_hmap_vis', img_hmap_final)
    cv2.waitKey(1)

    return
#end







########################################################################################################################
###
########################################################################################################################
def decode_segmap(labelmap, plot=False):
    ###===================================================================================================
    ### convert label_map into visible_img [only called in __main__()]
    ###===================================================================================================

    # labelmap: label_map, ndarray (360, 480)


    ###------------------------------------------------------------------------------------------
    ### setting
    ###------------------------------------------------------------------------------------------

    n_classes = 19

    ###
    rgb_class00 = [128,  64, 128]   # 00: road
    rgb_class01 = [244,  35, 232]   # 01: sidewalk
    rgb_class02 = [ 70,  70,  70]   # 02: construction
    rgb_class03 = [192,   0, 128]   # 03: tram-track
    rgb_class04 = [190, 153, 153]   # 04: fence
    rgb_class05 = [153, 153, 153]   # 05: pole
    rgb_class06 = [250, 170,  30]   # 06: traffic-light
    rgb_class07 = [220, 220,   0]   # 07: traffic-sign
    rgb_class08 = [107, 142,  35]   # 08: vegetation
    rgb_class09 = [152, 251, 152]   # 09: terrain
    rgb_class10 = [ 70, 130, 180]   # 10: sky
    rgb_class11 = [220,  20,  60]   # 11: human
    rgb_class12 = [230, 150, 140]   # 12: rail-track
    rgb_class13 = [  0,   0, 142]   # 13: car
    rgb_class14 = [  0,   0,  70]   # 14: truck
    rgb_class15 = [ 90,  40,  40]   # 15: trackbed
    rgb_class16 = [  0,  80, 100]   # 16: on-rails
    rgb_class17 = [  0, 254, 254]   # 17: rail-raised
    rgb_class18 = [  0,  68,  63]   # 18: rail-embedded


    ###
    rgb_labels = np.array(
        [
            rgb_class00,
            rgb_class01,
            rgb_class02,
            rgb_class03,
            rgb_class04,
            rgb_class05,
            rgb_class06,
            rgb_class07,
            rgb_class08,
            rgb_class09,
            rgb_class10,
            rgb_class11,
            rgb_class12,
            rgb_class13,
            rgb_class14,
            rgb_class15,
            rgb_class16,
            rgb_class17,
            rgb_class18,
        ]
    )


    ###------------------------------------------------------------------------------------------
    ### convert label_map into img_label_rgb
    ###------------------------------------------------------------------------------------------

    ### create default img
    r = np.ones_like(labelmap )*250          # 250: indicating invalid label
    g = np.ones_like(labelmap )*250
    b = np.ones_like(labelmap )*250

    for l in range(0, n_classes):
        ### find
        idx_set = (labelmap == l)           # idx_set: ndarray, (h, w), bool

        ### assign
        r[idx_set] = rgb_labels[l, 0]       # r: 0 ~ 255
        g[idx_set] = rgb_labels[l, 1]       # g: 0 ~ 255
        b[idx_set] = rgb_labels[l, 2]       # b: 0 ~ 255
    # end

    img_label_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3))
    img_label_rgb[:, :, 0] = r / 255.0
    img_label_rgb[:, :, 1] = g / 255.0
    img_label_rgb[:, :, 2] = b / 255.0

    return img_label_rgb
#end

########################################################################################################################
###
########################################################################################################################
def decode_segmap_b(labelmap):
    ###===================================================================================================
    ### convert label_map into visible_img [only called in __main__()]
    ###===================================================================================================

    # labelmap: label_map, ndarray (360, 480)


    ###------------------------------------------------------------------------------------------
    ### setting
    ###------------------------------------------------------------------------------------------
    n_classes = 19

    ###
    rgb_class00 = [128,  64, 128]   # 00: road
    rgb_class01 = [244,  35, 232]   # 01: sidewalk
    rgb_class02 = [ 70,  70,  70]   # 02: construction
    rgb_class03 = [192,   0, 128]   # 03: tram-track
    rgb_class04 = [190, 153, 153]   # 04: fence
    rgb_class05 = [153, 153, 153]   # 05: pole
    rgb_class06 = [250, 170,  30]   # 06: traffic-light
    rgb_class07 = [220, 220,   0]   # 07: traffic-sign
    rgb_class08 = [107, 142,  35]   # 08: vegetation
    rgb_class09 = [152, 251, 152]   # 09: terrain
    rgb_class10 = [ 70, 130, 180]   # 10: sky
    rgb_class11 = [220,  20,  60]   # 11: human
    rgb_class12 = [230, 150, 140]   # 12: rail-track
    rgb_class13 = [  0,   0, 142]   # 13: car
    rgb_class14 = [  0,   0,  70]   # 14: truck
    rgb_class15 = [ 90,  40,  40]   # 15: trackbed
    rgb_class16 = [  0,  80, 100]   # 16: on-rails
    rgb_class17 = [  0, 254, 254]   # 17: rail-raised
    rgb_class18 = [  0,  68,  63]   # 18: rail-embedded


    ###
    rgb_labels = np.array(
        [
            rgb_class00,
            rgb_class01,
            rgb_class02,
            rgb_class03,
            rgb_class04,
            rgb_class05,
            rgb_class06,
            rgb_class07,
            rgb_class08,
            rgb_class09,
            rgb_class10,
            rgb_class11,
            rgb_class12,
            rgb_class13,
            rgb_class14,
            rgb_class15,
            rgb_class16,
            rgb_class17,
            rgb_class18,
        ]
    )


    ###------------------------------------------------------------------------------------------
    ### convert label_map into img_label_rgb
    ###------------------------------------------------------------------------------------------

    ### create default img
    r = np.ones_like(labelmap )*250          # 250: indicating invalid label
    g = np.ones_like(labelmap )*250
    b = np.ones_like(labelmap )*250

    for l in range(0, n_classes):
        ### find
        idx_set = (labelmap == l)           # idx_set: ndarray, (h, w), bool

        ### assign
        r[idx_set] = rgb_labels[l, 0]       # r: 0 ~ 255
        g[idx_set] = rgb_labels[l, 1]       # g: 0 ~ 255
        b[idx_set] = rgb_labels[l, 2]       # b: 0 ~ 255
    # end

    img_label_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3))
    img_label_rgb[:, :, 0] = r
    img_label_rgb[:, :, 1] = g
    img_label_rgb[:, :, 2] = b



    img_label_rgb = img_label_rgb[:, :, ::-1]       # rgb -> bgr (for following opencv convention)
    img_label_rgb = img_label_rgb.astype(np.uint8)


    return img_label_rgb
#end



########################################################################################################################
###
########################################################################################################################
def decode_output_centerline(res_in):
    ###============================================================================================================
    ### visualize output_centerline
    ###============================================================================================================

    ###
    res_sigmoid = torch.clamp(torch.sigmoid(res_in), min=1e-4, max=1 - 1e-4)


    ###
    res_a = res_sigmoid[0].permute(1, 2, 0)     # res_a : tensor(512, 1024, 1)
    res_b = res_a[:, :, 0]                      # res_b : tensor(512, 1024)
    res_c = res_b * 255.0
    res_d = torch.clamp(res_c, min=0.0, max=255.0)
    res_e = res_d.detach().cpu().numpy()


    ###
    res_out = res_b.detach().cpu().numpy()
        # completed to set
        #       res_out: ndarray (h,w), 0.0 ~ 1.0

    ###
    img_res_out = res_e.astype(np.uint8)
        # completed to set
        #       img_res_out: ndarray (h,w), uint8


    return res_out, img_res_out
#end


########################################################################################################################
###
########################################################################################################################
def decode_output_leftright(res_in):
    ###============================================================================================================
    ### visualize output_leftright
    ###============================================================================================================

    ###
    res_relu = torch.relu(res_in)


    ###
    res0     = res_relu[0]                                          # res0: tensor (2, h, w)

    ###
    res_left_a      = res0[0, :, :]                                 # res_left  : tensor(h, w)
    res_left_b      = torch.clamp(res_left_a, min=0.0, max=255.0)
    res_left_c      = res_left_b.detach().cpu().numpy()
    img_res_left    = res_left_c.astype(np.uint8)
        # completed to set
        #       img_res_left: ndarray (h,w), uint8


    ###
    res_right_a     = res0[1, :, :]                                 # res_right : tensor(h, w)
    res_right_b     = torch.clamp(res_right_a, min=0.0, max=255.0)
    res_right_c     = res_right_b.detach().cpu().numpy()
    img_res_right   = res_right_c.astype(np.uint8)
        # completed to set
        #       img_res_right: ndarray (h,w), uint8


    ###
    res_left = res_left_c
    res_right = res_right_c
        # completed to set
        #       res_left: ndarray (h,w) float32     0.0 ~ X.X
        #       res_right: ndarray (h,w) float32    0.0 ~ X.X


    return res_left, res_right, img_res_left, img_res_right
#end


########################################################################################################################
###
########################################################################################################################
def visualize_res_temp0(img_raw_rsz_uint8, res_centerline, res_left, res_right):
    ###=========================================================================================================
    ### show centerline and corresponding left, right rails
    ###
    ### res_centerline: 0.0 ~ 1.0, float32
    ### res_left:       0.0 ~ X.X, float32
    ### res_right:      0.0 ~ X.X, float32
    ###=========================================================================================================


    ###================================================================================================
    ###
    ###================================================================================================
    h_img = res_centerline.shape[0]
    w_img = res_centerline.shape[1]

    ###
    yx_center = np.where(res_centerline >= 0.7)

    set_y_center = yx_center[0]
    set_x_center = yx_center[1]

    totnum_yx = set_y_center.shape[0]
        # completed to set
        #       set_y_center
        #       set_x_center
        #       totnum_yx


    ###================================================================================================
    ###
    ###================================================================================================

    ###
    img_res = np.zeros(shape=(h_img, w_img, 3), dtype=np.uint8)     # RGB


    ### left
    for idx_yx in range(0, totnum_yx):
        y_cen = set_y_center[idx_yx]
        x_cen = set_x_center[idx_yx]

        dx = res_left[y_cen, x_cen]

        y_left = y_cen
        x_left = max(0, int(round(x_cen - dx)))

        img_res[y_left, x_left, 0] = 255            # R
        img_res[y_left, x_left, 1] = 0
        img_res[y_left, x_left, 2] = 0
    #end


    ### right
    for idx_yx in range(0, totnum_yx):
        y_cen = set_y_center[idx_yx]
        x_cen = set_x_center[idx_yx]

        dx = res_right[y_cen, x_cen]

        y_left = y_cen
        x_left = min(int(round(x_cen + dx)), w_img - 1)

        img_res[y_left, x_left, 0] = 0
        img_res[y_left, x_left, 1] = 0
        img_res[y_left, x_left, 2] = 255
    #end


    ### center
    for idx_yx in range(0, totnum_yx):
        y_cen = set_y_center[idx_yx]
        x_cen = set_x_center[idx_yx]

        centerness = res_centerline[y_cen, x_cen]       # 0.0 ~ 1.0
        val_pixel  = round(centerness*255.0)

        if val_pixel < 0.0:
            val_pixel = 0.0
        #end

        if val_pixel > 255.0:
            val_pixel = 255.0
        #end

        img_res[y_cen, x_cen, :] = np.uint8(val_pixel)
    #end
        # completed to set
        #       img_res


    ###================================================================================================
    ###
    ###================================================================================================
    img_res_vis       = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
    img_res_vis_final = cv2.addWeighted(src1=img_raw_rsz_uint8, alpha=0.25, src2=img_res_vis, beta=0.75, gamma=0)


    ###
    # cv2.imshow('img_res_vis_final', img_res_vis_final)
    # cv2.waitKey(1)


    return img_res_vis_final
#end


########################################################################################################################
###
########################################################################################################################
def visualize_res_temp1(img_raw_rsz_uint8, res_left, res_right):
    ###=========================================================================================================
    ### show left rails - right rails
    ###
    ### res_left:       0.0 ~ X.X, float32
    ### res_right:      0.0 ~ X.X, float32
    ###=========================================================================================================


    ###================================================================================================
    ###
    ###================================================================================================
    res_delta = abs(res_left - res_right)

    res_delta[res_delta > 255.0] = 255.0

    img_res = res_delta.astype(np.uint8)


    ###================================================================================================
    ###
    ###================================================================================================
    img_res_vis       = cv2.cvtColor(img_res, cv2.COLOR_GRAY2BGR)
    #img_res_vis_final = cv2.addWeighted(src1=img_raw_rsz_uint8, alpha=0.25, src2=img_res_vis, beta=0.75, gamma=0)


    return img_res_vis
#end


########################################################################################################################
### adjust rgb
########################################################################################################################
def adjust_rgb(type, b_old_uint8, g_old_uint8, r_old_uint8):

    ###
    dr_int = 0
    dg_int = 0
    db_int = 0

    if type == 0:       # track region
        dr_int = 0
        dg_int = 100
        db_int = 0
    elif type == 1:     # left
        dr_int = 100
        dg_int = 0
        db_int = 0
    elif type == 2:     # right
        dr_int = 0
        dg_int = 0
        db_int = 100
    elif type == 3:     # center
        dr_int = 0
        dg_int = 200
        db_int = 0
    #end


    ###
    r_new_int = int(r_old_uint8) + dr_int
    g_new_int = int(g_old_uint8) + dg_int
    b_new_int = int(b_old_uint8) + db_int


    ###
    r_new_int = min(r_new_int, 255)
    r_new_int = max(r_new_int, 0)

    g_new_int = min(g_new_int, 255)
    g_new_int = max(g_new_int, 0)

    b_new_int = min(b_new_int, 255)
    b_new_int = max(b_new_int, 0)


    ###
    return b_new_int, g_new_int, r_new_int
#end

########################################################################################################################
###
########################################################################################################################
def visualize_res_temp2(img_raw_rsz_uint8, res_centerness, res_left, res_right):
    ###=========================================================================================================
    ### show centerline and corresponding left, right rails
    ###
    ### res_centerness: 0.0 ~ 1.0, float32
    ### res_left:       0.0 ~ X.X, float32
    ### res_right:      0.0 ~ X.X, float32
    ###=========================================================================================================

    # <scipy.signal.find_peaks>
    #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    #   https://stackoverflow.com/questions/31070563/find-all-local-maxima-and-minima-when-x-and-y-values-are-given-as-numpy-arrays/31073798


    h_img = res_centerness.shape[0]
    w_img = res_centerness.shape[1]


    ###================================================================================================
    ### find local maxima
    ###================================================================================================
    mat_local_max = np.zeros_like(res_centerness)

    for y in range(0, h_img):
        ###
        centerness_thisrow = res_centerness[y, :]

        ###
        dist_min = (235.0/270.0)*y - 220.0
        dist_min = max(15.0, dist_min)
        dist_min = dist_min*0.6

        ###
        set_x_peaks, _ = find_peaks(centerness_thisrow, height=0.5, distance=dist_min)

        ###
        num_local_max = set_x_peaks.size

        for i in range(0, num_local_max):
            x_this = set_x_peaks[i]
            c_this = centerness_thisrow[x_this]
            mat_local_max[y, x_this] = c_this
        #end
    #end
        # completed to set
        #       mat_local_max: ndarray (h,w), 0.0 ~ 1.0


    ### <debugging>>
    # img_mat_local_max = mat_local_max*255.0
    # img_mat_local_max = img_mat_local_max.astype(np.uint8)
    # img_mat_local_max_rgb = cv2.cvtColor(img_mat_local_max, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('img_mat_local_max_rgb', img_mat_local_max_rgb)
    # cv2.waitKey(1)


    ###================================================================================================
    ###
    ###================================================================================================

    ###
    yx_center = np.where(mat_local_max >= 0.5)

    set_y_center = yx_center[0]
    set_x_center = yx_center[1]

    totnum_yx = set_y_center.shape[0]
        # completed to set
        #       set_y_center
        #       set_x_center
        #       totnum_yx


    ###================================================================================================
    ###
    ###================================================================================================
    img_res_rgb = copy.deepcopy(img_raw_rsz_uint8)

    ###
    for idx_yx in range(0, totnum_yx):
        y_cen = set_y_center[idx_yx]
        x_cen = set_x_center[idx_yx]

        dx_left  = res_left [y_cen, x_cen]
        dx_right = res_right[y_cen, x_cen]

        y_this  = y_cen
        x_left  = max(0, int(round(x_cen - dx_left)))
        x_right = min(int(round(x_cen + dx_right)), w_img - 1)

        ### fill region
        for x_this in range(x_left, x_right + 1):
            b_old, g_old, r_old = img_res_rgb[y_this, x_this, :]
            b_new, g_new, r_new = adjust_rgb(0, b_old, g_old, r_old)
            img_res_rgb[y_this, x_this, :] = (b_new, g_new, r_new)
        #end
    #end
        # completed to set
        #       img_res_rgb



    ###
    for idx_yx in range(0, totnum_yx):
        y_cen = set_y_center[idx_yx]
        x_cen = set_x_center[idx_yx]

        dx_left  = res_left [y_cen, x_cen]
        dx_right = res_right[y_cen, x_cen]

        y_this  = y_cen
        x_left  = max(0, int(round(x_cen - dx_left)))
        x_right = min(int(round(x_cen + dx_right)), w_img - 1)

        ### draw pnts
        cv2.circle(img_res_rgb, center=(x_left, y_this),  radius=3, color=(20,  100, 250), thickness=-1)
        cv2.circle(img_res_rgb, center=(x_right, y_this), radius=3, color=(250, 250,   0), thickness=-1)
        cv2.circle(img_res_rgb, center=(x_cen, y_this),   radius=3, color=(0,   128,   0), thickness=-1)
    #end
        # completed to set
        #       img_res_rgb




    ###================================================================================================
    ###
    ###================================================================================================
    # alpha = 0.3
    # beta  = 1.0 - alpha
    # img_res_final = cv2.addWeighted(src1=img_res_region_rgb, alpha=alpha, src2=img_res_pnt_rgb, beta=beta, gamma=0)


    ###================================================================================================
    ###
    ###================================================================================================
    cv2.imshow('visualize_res_temp2', img_res_rgb)
    cv2.waitKey(1)


    return img_res_rgb
#end



########################################################################################################################
###
########################################################################################################################
def compute_centerness_from_leftright(res_left, res_right):
    ###=========================================================================================================
    ### show left rails - right rails
    ###
    ### res_left:       0.0 ~ X.X, float32
    ### res_right:      0.0 ~ X.X, float32
    ###=========================================================================================================


    ###================================================================================================
    ###
    ###================================================================================================
    res_delta = abs(res_left - res_right)
    res_sum   = abs(res_left) + abs(res_right)


    res_ratio = res_delta/res_sum           # close to 0: high centerness

    res_ratio[np.isnan(res_ratio)] = 1.0
    res_ratio[res_sum <= 1.0] = 1.0

    res_weight = 1.0 - res_ratio


    res_weight_b   = res_weight*255.0
    img_res_weight = res_weight_b.astype(np.uint8)


    return res_weight, img_res_weight
#end





########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


    # res_temp = res_centerline*255.0
    # res_temp = res_temp.astype(np.uint8)
    # img_res  = cv2.cvtColor(res_temp, cv2.COLOR_GRAY2BGR)
    #
    # ###
    # cv2.imshow('img_vis_res', img_res)
    # cv2.waitKey(1)


"""
########################################################################################################################
###
########################################################################################################################
def visualize_res(img_raw_rsz_uint8, res_centerline, res_left, res_right):
    ###=========================================================================================================
    ###
    ### res_centerline: 0.0 ~ 1.0, float32
    ### res_left:       0.0 ~ X.X, float32
    ### res_right:      0.0 ~ X.X, float32
    ###=========================================================================================================


    ###
    set_idx_center = res_centerline >= 0.90
    set_idx_center_non = ~set_idx_center

    ###
    res_centerline[set_idx_center_non] = 0.0
    res_centerline[set_idx_center] = 1.0



    # res_temp = res_centerline*255.0
    # res_temp = res_temp.astype(np.uint8)
    # img_res  = cv2.cvtColor(res_temp, cv2.COLOR_GRAY2BGR)
    #
    # ###
    # cv2.imshow('img_vis_res', img_res)
    # cv2.waitKey(1)



    ###
    h_img = res_centerline.shape[0]
    w_img = res_centerline.shape[1]

    img_res = np.zeros(shape=(h_img, w_img, 3), dtype=np.uint8)     # RGB


    ###
    yx_center = np.where(res_centerline >= 0.5)

    set_y_center = yx_center[0]
    set_x_center = yx_center[1]

    totnum_yx = set_y_center.shape[0]


    # ### left
    # for idx_yx in range(0, totnum_yx + 1):
    #     y_cen = set_y_center[idx_yx]
    #     x_cen = set_x_center[idx_yx]
    #
    #     dx = res_left[y_cen, x_cen]
    #
    #     a = 1
    #     # img_res[y_cen, x_cen,
    # #end


    ### left
    for idx_yx in range(0, totnum_yx):
        y_cen = set_y_center[idx_yx]
        x_cen = set_x_center[idx_yx]

        dx = res_left[y_cen, x_cen]

        y_left = y_cen
        x_left = max(0, int(round(x_cen - dx)))

        img_res[y_left, x_left, 0] = 255
        img_res[y_left, x_left, 1] = 0
        img_res[y_left, x_left, 2] = 0
    #end


    ### right
    for idx_yx in range(0, totnum_yx):
        y_cen = set_y_center[idx_yx]
        x_cen = set_x_center[idx_yx]

        dx = res_right[y_cen, x_cen]

        y_left = y_cen
        x_left = min(int(round(x_cen + dx)), w_img - 1)

        img_res[y_left, x_left, 0] = 0
        img_res[y_left, x_left, 1] = 0
        img_res[y_left, x_left, 2] = 255
    #end


    ### center
    for idx_yx in range(0, totnum_yx):
        y_cen = set_y_center[idx_yx]
        x_cen = set_x_center[idx_yx]

        img_res[y_cen, x_cen, :] = 255
    #end


    img_res_vis  = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
    img_res_final = cv2.addWeighted(src1=img_raw_rsz_uint8, alpha=0.25, src2=img_res_vis, beta=0.75, gamma=0)


    ###
    cv2.imshow('img_vis_res_final', img_res_final)
    cv2.waitKey(1)


    return
#end
"""

# ### right
# for idx_yx in range(0, totnum_yx):
#     y_cen = set_y_center[idx_yx]
#     x_cen = set_x_center[idx_yx]
#
#     dx = res_right[y_cen, x_cen]
#
#     y_left = y_cen
#     x_left = min(int(round(x_cen + dx)), w_img - 1)
#
#     # img_res[y_left, x_left, 0] = 0
#     # img_res[y_left, x_left, 1] = 0
#     # img_res[y_left, x_left, 2] = 255
# #end


# ### center
# for idx_yx in range(0, totnum_yx):
#     y_cen = set_y_center[idx_yx]
#     x_cen = set_x_center[idx_yx]
#
#     centerness = res_centerness[y_cen, x_cen]       # 0.0 ~ 1.0
#     val_pixel  = round(centerness*255.0)
#
#     val_pixel  = max(0.0, val_pixel)
#     val_pixel  = min(val_pixel, 255.0)
#
#     # img_res[y_cen, x_cen, :] = np.uint8(val_pixel)
# #end
# completed to set
#       img_res


# ### draw pnts (left)
# b_ori, g_ori, r_ori = img_raw_rsz_uint8[y_this, x_left, :]
# b_new, g_new, r_new = adjust_rgb(1, b_ori, g_ori, r_ori)
# cv2.circle(img_res_region_rgb, center=(x_left, y_this), radius=3, color=(b_new, g_new, r_new), thickness=-1)
#
# ### draw pnts (right)
# b_ori, g_ori, r_ori = img_raw_rsz_uint8[y_this, x_right, :]
# b_new, g_new, r_new = adjust_rgb(2, b_ori, g_ori, r_ori)
# cv2.circle(img_res_region_rgb, center=(x_right, y_this), radius=3, color=(b_new, g_new, r_new), thickness=-1)
#
# ### draw pnts (center)
# b_ori, g_ori, r_ori = img_raw_rsz_uint8[y_this, x_cen, :]
# b_new, g_new, r_new = adjust_rgb(3, b_ori, g_ori, r_ori)
# cv2.circle(img_res_region_rgb, center=(x_cen, y_this), radius=3, color=(b_new, g_new, r_new), thickness=-1)
