# 2020/8/11
# Jungwon Kang


import os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import copy



from ptsemseg.models            import get_model
from ptsemseg.loader            import get_loader
from ptsemseg.utils             import convert_state_dict
from ptsemseg.loader.myhelpers  import myhelper_railsem19_b
from helpers_my                 import my_utils
from helpers_my                 import my_util_etc

torch.backends.cudnn.benchmark = True
import cv2



########################################################################################################################
### class PathExtraction3D_TPE_Net
########################################################################################################################
class PathExtraction3D_TPE_Net:

    ###
    m_device           = None   # gpu
    m_model            = None   # net


    ###=========================================================================================================
    ### __init__()
    ###=========================================================================================================
    def __init__(self, args):

        ###---------------------------------------------------------------------------------------------
        ### init dataloader
        ###---------------------------------------------------------------------------------------------
        data_loader = get_loader(args.dataset)
        loader      = data_loader(output_size_hmap="size_img_rsz")
        n_classes   = loader.n_classes
            # completed to set
            #       loader
            #       n_classes


        ###---------------------------------------------------------------------------------------------
        ### init model
        ###---------------------------------------------------------------------------------------------
        self.m_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ###
        self.m_model = get_model(args.name_model, n_classes, version=args.dataset)

        ###
        my_utils.load_weights_to_model(self.m_model, args.file_weight)

        self.m_model.eval()
        self.m_model.to(self.m_device)
            # completed to set
            #       m_device
            #       m_model

        return
    #end


    ###=========================================================================================================
    ### process()
    ###=========================================================================================================
    def process(self, img_raw_rsz_uint8, img_raw_rsz_fl_n):
        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        img_raw = np.expand_dims(img_raw_rsz_fl_n, 0)
        img_raw = torch.from_numpy(img_raw).float()

        images = img_raw.to(self.m_device)


        ###------------------------------------------------------------------------------------------------
        ### feed-forwarding
        ###------------------------------------------------------------------------------------------------
        output_seg, \
        output_centerline, \
        output_leftright = self.m_model(images)
            # At this point, completed to set
            #       output_seg
            #       output_centerline
            #       output_leftright


        ###------------------------------------------------------------------------------------------------
        ### decode network_output (& create img for visualization)
        ###------------------------------------------------------------------------------------------------

        ### decode output_seg
        labels_seg_predicted = np.squeeze(output_seg.data.max(1)[1].cpu().numpy(), axis=0)
        img_res_seg = myhelper_railsem19_b.decode_segmap_b(labels_seg_predicted)
            # completed to set
            #       img_res_seg


        ### decode output_centerline
        res_centerness_direct, \
        img_res_centerness_direct = myhelper_railsem19_b.decode_output_centerline(output_centerline)
            # completed to set
            #       res_centerness_direct
            #       img_res_centerness_direct


        ### decode output_leftright
        res_left, \
        res_right, \
        img_res_left, \
        img_res_right = myhelper_railsem19_b.decode_output_leftright(output_leftright)
            # completed to set
            #       res_left, res_right
            #       img_res_left, img_res_right


        # At this point, completed to set
        #       img_res_seg
        #       res_centerness_direct, img_res_centerness_direct
        #       img_res_left, img_res_right, res_left, res_right


        ###------------------------------------------------------------------------------------------------
        ### process (for combining centerness-direct and centerness-from-left-right)
        ###------------------------------------------------------------------------------------------------
        res_centerness_from_LR,\
        img_res_centerness_from_LR = myhelper_railsem19_b.compute_centerness_from_leftright(res_left, res_right)
            # completed to set
            #       res_centerness_from_LR
            #       img_res_centerness_from_LR


        ###
        res_centerness_combined = res_centerness_direct*res_centerness_from_LR
        res_centerness_combined_b = res_centerness_combined*255.0
        img_res_centerness_combined = res_centerness_combined_b.astype(np.uint8)


        ###------------------------------------------------------------------------------------------------
        ### visualize result
        ###------------------------------------------------------------------------------------------------
        img_res_vis_temp0 = myhelper_railsem19_b.visualize_res_temp0(img_raw_rsz_uint8, res_centerness_direct, res_left, res_right)
        img_res_vis_temp1 = myhelper_railsem19_b.visualize_res_temp1(img_raw_rsz_uint8, res_left, res_right)
        img_res_vis_temp2 = myhelper_railsem19_b.visualize_res_temp2(img_raw_rsz_uint8, res_centerness_combined, res_left, res_right)



        return  img_raw_rsz_uint8, \
                img_res_seg, \
                img_res_centerness_direct, \
                img_res_left, \
                img_res_right, \
                img_res_centerness_from_LR, \
                img_res_centerness_combined, \
                img_res_vis_temp0, \
                img_res_vis_temp1, \
                img_res_vis_temp2
    #end


#end



########################################################################################################################
### __main__()
########################################################################################################################
if __name__ == "__main__":
    ###============================================================================================================
    ### (1) set parser
    ###============================================================================================================
    parser = argparse.ArgumentParser(description="Params")
    args = parser.parse_args()

    ### put parameters into args
    args.file_weight = "/home/yu1/PycharmProjects/proj_seg_hardnet_a/train_py/runs/rpnet_c_railsem19_seg_triplet0/cur_20200725/rpnet_c_railsem19_seg_triplet_b_best_model_20800.pkl"
    args.name_model  = {"arch": "rpnet_c"}
    args.dataset     = "railsem19_seg_triplet_b"
    args.size_img_process = {'h': 540, 'w': 960}
    args.dir_input   = "/home/yu1/proj_avin/dataset/img_nyc_1280_720"
    args.dir_output  = "/home/yu1/Desktop/dir_temp/temp5"



    ###============================================================================================================
    ### (2) init
    ###============================================================================================================
    PathExtractor = PathExtraction3D_TPE_Net(args)



    ###============================================================================================================
    ### (3) loop
    ###============================================================================================================
    print("Process all image inside : {}".format(args.dir_input))

    list_fnames_img_ = os.listdir(args.dir_input)
    list_fnames_img  = sorted(list_fnames_img_)


    for fname_img_in in list_fnames_img:
        ###------------------------------------------------------------------------------------------------
        ### read img from file
        ###------------------------------------------------------------------------------------------------
        full_fname_img_ori = os.path.join(args.dir_input, fname_img_in)

        print("Read Input Image from : {}".format(full_fname_img_ori))


        img_raw_rsz_uint8, \
        img_raw_rsz_fl_n = myhelper_railsem19_b.read_img_raw_jpg_from_file(full_fname_img_ori, args.size_img_process)



        ###------------------------------------------------------------------------------------------------
        ### get result
        ###------------------------------------------------------------------------------------------------
        img_raw_in, \
        img_res_seg, \
        img_res_centerness_direct, \
        img_res_left, \
        img_res_right, \
        img_res_centerness_from_LR, \
        img_res_centerness_combined, \
        img_res_vis_temp0, \
        img_res_vis_temp1, \
        img_res_vis_temp2 = PathExtractor.process(img_raw_rsz_uint8, img_raw_rsz_fl_n)


        ###------------------------------------------------------------------------------------------------
        ### set fname (for saving)
        ###------------------------------------------------------------------------------------------------

        ###
        fname_out_img_raw_in                    = args.dir_output + '/in/'                  + 'in_'                 + fname_img_in
        fname_out_img_res_seg                   = args.dir_output + '/seg/'                 + 'seg_'                + fname_img_in
        fname_out_img_res_centerness_direct     = args.dir_output + '/center_direct/'       + 'center_direct_'      + fname_img_in
        fname_out_img_res_left                  = args.dir_output + '/left/'                + 'left_'               + fname_img_in
        fname_out_img_res_right                 = args.dir_output + '/right/'               + 'right_'              + fname_img_in
        fname_out_img_res_centerness_from_LR    = args.dir_output + '/center_LR/'           + 'center_LR_'          + fname_img_in
        fname_out_img_res_centerness_combined   = args.dir_output + '/center_combined/'     + 'center_combined_'    + fname_img_in
        fname_out_img_res_vis_temp0             = args.dir_output + '/vis_temp0/'           + 'vis_temp0_'          + fname_img_in
        fname_out_img_res_vis_temp1             = args.dir_output + '/vis_temp1/'           + 'vis_temp1_'          + fname_img_in
        fname_out_img_res_vis_temp2             = args.dir_output + '/vis_temp2/'           + 'vis_temp2_'          + fname_img_in


        ###
        fname_out_img_raw_in_bev                    = args.dir_output + '/in_bev/'              + 'in_bev_'             + fname_img_in
        fname_out_img_res_seg_bev                   = args.dir_output + '/seg_bev/'             + 'seg_bev_'            + fname_img_in
        fname_out_img_res_centerness_direct_bev     = args.dir_output + '/center_direct_bev/'   + 'center_direct_bev_'  + fname_img_in
        fname_out_img_res_centerness_combined_bev   = args.dir_output + '/center_combined_bev/' + 'center_combined_bev' + fname_img_in
        fname_out_img_res_vis_temp2_bev             = args.dir_output + '/vis_temp2_bev/'       + 'vis_temp2_bev'       + fname_img_in



        ###------------------------------------------------------------------------------------------------
        ### convert img
        ###------------------------------------------------------------------------------------------------
        img_res_centerness_direct_rgb   = cv2.cvtColor(img_res_centerness_direct,   cv2.COLOR_GRAY2BGR)
        img_res_left_rgb                = cv2.cvtColor(img_res_left,                cv2.COLOR_GRAY2BGR)
        img_res_right_rgb               = cv2.cvtColor(img_res_right,               cv2.COLOR_GRAY2BGR)
        img_res_centerness_from_LR_rgb  = cv2.cvtColor(img_res_centerness_from_LR,  cv2.COLOR_GRAY2BGR)
        img_res_centerness_combined_rgb = cv2.cvtColor(img_res_centerness_combined, cv2.COLOR_GRAY2BGR)


        ###------------------------------------------------------------------------------------------------
        ### show and save (2d)
        ###------------------------------------------------------------------------------------------------
        if 1:
            ###
            cv2.imshow('img_raw_in', img_raw_in)
            cv2.imwrite(fname_out_img_raw_in, img_raw_in)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_seg', img_res_seg)
            cv2.imwrite(fname_out_img_res_seg, img_res_seg)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_centerness_direct', img_res_centerness_direct_rgb)
            cv2.imwrite(fname_out_img_res_centerness_direct, img_res_centerness_direct_rgb)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_left', img_res_left_rgb)
            cv2.imwrite(fname_out_img_res_left, img_res_left_rgb)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_right', img_res_right_rgb)
            cv2.imwrite(fname_out_img_res_right, img_res_right_rgb)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_centerness_from_LR', img_res_centerness_from_LR)            # from left & right
            cv2.imwrite(fname_out_img_res_centerness_from_LR, img_res_centerness_from_LR)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_centerness_combined', img_res_centerness_combined)          # combination of direct and LR
            cv2.imwrite(fname_out_img_res_centerness_combined, img_res_centerness_combined)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_vis_temp0', img_res_vis_temp0)                              # center, and corresponding left, right
            cv2.imwrite(fname_out_img_res_vis_temp0, img_res_vis_temp0)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_vis_temp1', img_res_vis_temp1)                              # left - right
            cv2.imwrite(fname_out_img_res_vis_temp1, img_res_vis_temp1)
            cv2.waitKey(1)

            ###
            cv2.imshow('img_res_vis_temp2', img_res_vis_temp2)                              # center, and corresponding left, right
            cv2.imwrite(fname_out_img_res_vis_temp2, img_res_vis_temp2)
            cv2.waitKey(1)
        #end



    #end
#end


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

