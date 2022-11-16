# 2020/7/10
# Jungwon Kang


import os
import torch
import numpy as np
import collections
import cv2
import math
import copy

from ptsemseg.loader.myhelpers.lib.image import draw_umich_gaussian, gaussian_radius



########################################################################################################################
###
########################################################################################################################
def read_img_raw_jpg_from_file(full_fname_img_raw_jpg, size_img_rsz,
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
    img_raw_rsz_fl_n_final = convert_img_ori_to_img_data(img_raw_rsz_uint8)
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


    ###================================================================================================
    ### (1) normalize so that 0~255 -> 0.0~1.0
    ###================================================================================================
    img_ori_fl = img_ori_uint8.astype(np.float32) / 255.0


    ###================================================================================================
    ### (2) apply rgb_mean
    ###================================================================================================
    img_ori_fl_n = img_ori_fl - rgb_mean
        # completed to set
        #       img_ori_fl_n: -X.0 ~ X.0, ndarray(H,W,C)


    ###================================================================================================
    ### (3) apply rgb_std
    ###================================================================================================
    img_ori_fl_n = img_ori_fl_n / rgb_std


    ###================================================================================================
    ### (4) convert HWC -> CHW
    ###================================================================================================
    img_ori_fl_n = img_ori_fl_n.transpose(2, 0, 1)
        # H(0),W(1),C(2) -> C(2),H(0),W(1)
        # completed to set
        #       img_ori_fl_n: -1.0 ~ 1.0, ndarray(C,H,W)


    ###================================================================================================
    ### (5) make sure it is float32 type
    ###================================================================================================
    img_data_fl_n_final = img_ori_fl_n.astype(np.float32)


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
def create_hmap_fmap(list_triplet_json, size_fmap, FACTOR_ori_to_fmap_h, FACTOR_ori_to_fmap_w):
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
    gaussian_iou = 0.7


    ###================================================================================================
    ###
    ###================================================================================================
    hmap = np.zeros((num_classes, size_fmap['h'], size_fmap['w']), dtype=np.float32)


    ###================================================================================================
    ###
    ###================================================================================================
    param_width_min = 10


    for list_this_set in list_triplet_json:
        # list_this_set: {list: N}, consists of N triplets.

        for triplet_this in list_this_set:
            # triplet_this: {list:4} (x_L, x_C, x_R, y)


            ### get this triplet
            x_L = int(round(triplet_this[0]*FACTOR_ori_to_fmap_w))
            x_C = int(round(triplet_this[1]*FACTOR_ori_to_fmap_w))
            x_R = int(round(triplet_this[2]*FACTOR_ori_to_fmap_w))
            y   = int(round(triplet_this[3]*FACTOR_ori_to_fmap_h))


            ### compute params for vote
            width_rail_ = x_R - x_L
            width_rail  = max(param_width_min, width_rail_)*1.5
            radius = max(5, int(gaussian_radius((math.ceil(width_rail), math.ceil(width_rail)), gaussian_iou)))
                # completed to set
                #       radius


            ### vote
            pnt_center = [x_C, y]
            draw_umich_gaussian(hmap[0], pnt_center, radius)
        # end
    # end
        # completed to set
        #       hmap


    return hmap
#end


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
def visualize_hmap(hmap_this):
    #===================================================================================================
    # visualize hmap
    # <input>
    #  hmap_this: ndarr(h, w), 0.0 ~ 1.0
    #===================================================================================================

    img_hmap_ = hmap_this*255.0
    img_hmap  = img_hmap_.astype(np.uint8)


    cv2.imshow('img_hmap', img_hmap)
    cv2.waitKey(1)

    return
#end


########################################################################################################################
###
########################################################################################################################
def visualize_hmap_b(hmap_this, img_raw_in):
    #===================================================================================================
    # visualize hmap
    # <input>
    #  hmap_this: ndarr(h, w), 0.0 ~ 1.0
    #===================================================================================================

    ### temp-rountine
    inds_pos = hmap_this >= 0.5
    inds_neg = hmap_this < 0.5

    hmap_this[inds_pos] = 1.0
    hmap_this[inds_neg] = 0.0



    img_hmap_ = hmap_this*255.0
    img_hmap  = img_hmap_.astype(np.uint8)
    img_hmap_rgb = cv2.cvtColor(img_hmap, cv2.COLOR_GRAY2BGR)


    img_hmap_final = cv2.addWeighted(src1=img_raw_in, alpha=0.25, src2=img_hmap_rgb, beta=0.75, gamma=0)
    cv2.imshow('img_hmap_b', img_hmap_final)
    #cv2.imwrite(fname_out_img, img_hmap_rgb)
    cv2.waitKey(1)

    return img_hmap_rgb
#end



########################################################################################################################
###
########################################################################################################################
def decode_output_centerline(res_in):
    ###============================================================================================================
    ### visualize outputs_centerline
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
    img_res_out = res_e.astype(np.uint8)
        # completed to set
        #       img_res_out: ndarray (h,w), uint8


    return img_res_out
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
    img_label_rgb.astype(np.uint8)


    return img_label_rgb
#end
########################################################################################################################
