# 2020/7/15
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
###
########################################################################################################################
def init_dataloader(args):

    ###
    data_loader = get_loader(args.dataset)
    loader      = data_loader(output_size_hmap="size_img_rsz")
    n_classes   = loader.n_classes

    return loader, n_classes
#end


########################################################################################################################
###
########################################################################################################################
def init_model_a(args, n_classes):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # see also <models/__init__.py>
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ### setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###
    model = get_model(args.name_model, n_classes, version=args.dataset)
    state_loaded = torch.load(args.file_weight)["model_state"]
    state = convert_state_dict(state_loaded)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return device, model
#end


########################################################################################################################
###
########################################################################################################################
def init_model_b(args, n_classes):
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # see also <models/__init__.py>
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ### setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ###
    model = get_model(args.name_model, n_classes, version=args.dataset)

    ###
    my_utils.load_weights_to_model(model, args.file_weight)

    model.eval()
    model.to(device)

    return device, model
#end


########################################################################################################################
###
########################################################################################################################
def process_img_my(full_fname_img_raw_jpg, size, device, model, loader, param_name_model):
    ###
    print("Read Input Image from : {}".format(full_fname_img_raw_jpg))


    ###=========================================================================================================
    ###
    ###=========================================================================================================
    img_raw_rsz_uint8, \
    img_raw_rsz_fl_n = myhelper_railsem19_b.read_img_raw_jpg_from_file(full_fname_img_raw_jpg,
                                                                       loader.size_img_rsz)


    img_raw = np.expand_dims(img_raw_rsz_fl_n, 0)
    img_raw = torch.from_numpy(img_raw).float()

    images = img_raw.to(device)


    ###=========================================================================================================
    ### feed-forwarding
    ###=========================================================================================================
    outputs_seg = None
    outputs_centerline = None
    outputs_leftright = None

    if param_name_model["arch"] is "rpnet_c":
        output_seg, \
        output_centerline, \
        output_leftright = model(images)
    #end

    # At this point, completed to set
    #       output_seg
    #       output_centerline
    #       output_leftright


    ###=========================================================================================================
    ### decode network_output (& create img for visualization)
    ###=========================================================================================================

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


    ###=========================================================================================================
    ### process (for combining centerness-direct and centerness-from-left-right)
    ###=========================================================================================================
    res_centerness_from_LR,\
    img_res_centerness_from_LR = myhelper_railsem19_b.compute_centerness_from_leftright(res_left, res_right)
        # completed to set
        #       res_centerness_from_LR
        #       img_res_centerness_from_LR


    ###
    res_centerness_combined = res_centerness_direct*res_centerness_from_LR
    res_centerness_combined_b = res_centerness_combined*255.0
    img_res_centerness_combined = res_centerness_combined_b.astype(np.uint8)


    ###=========================================================================================================
    ### visualize result
    ###=========================================================================================================
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




########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
### __main__()
########################################################################################################################
if __name__ == "__main__":
    ###============================================================================================================
    ### (1) set parameters
    ###============================================================================================================
    #param_file_weight = "/home/yu1/PycharmProjects/proj_seg_hardnet_a/train_py/runs/rpnet_c_railsem19_seg_triplet0/cur_20200722/rpnet_c_railsem19_seg_triplet_b_best_model_4100.pkl"
    param_file_weight = "/home/yu1/PycharmProjects/proj_seg_hardnet_a/train_py/runs/rpnet_c_railsem19_seg_triplet0/cur_20200725/rpnet_c_railsem19_seg_triplet_b_best_model_20800.pkl"
    param_name_model = {"arch": "rpnet_c"}
        # please, see <ptsemseg.models.__init__.py>
    param_dataset     = "railsem19_seg_triplet_b"
        # please, see <ptsemseg.loader.__init__.py>
    param_size        = "540,960"
    param_dir_input   = "/media/yu1/hdd_my/Dataset_YDHR_OCT009_OCT10/img_gopro/selected/test7_run2_normal_reverse_switch"
    param_dir_output  = "/home/yu1/Desktop/dir_temp/temp3a"


    ###============================================================================================================
    ### (2) set parser
    ###============================================================================================================
    parser = argparse.ArgumentParser(description="Params")
    args = parser.parse_args()

    ### put parameters into args
    args.file_weight = param_file_weight
    args.name_model  = param_name_model
    args.dataset     = param_dataset
    args.size        = param_size
    args.dir_input   = param_dir_input
    args.dir_output  = param_dir_output



    ###============================================================================================================
    ### (3) init
    ###============================================================================================================
    loader, n_classes = init_dataloader(args)
    device, model     = init_model_b(args, n_classes)
    proc_size         = eval(args.size)


    ###============================================================================================================
    ### (4) ipm
    ###============================================================================================================
    tf_ipm, param_h_img_bev, param_w_img_bev = my_util_etc.get_data_for_ipm()



    ###============================================================================================================
    ### (5) loop
    ###============================================================================================================
    print("Process all image inside : {}".format(args.dir_input))

    list_fnames_img_ = os.listdir(args.dir_input)
    list_fnames_img  = sorted(list_fnames_img_)


    for fname_img_in in list_fnames_img:
        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        _, ext = os.path.splitext(os.path.basename((fname_img_in)))
        if ext not in [".png", ".jpg"]:
            continue
        #end


        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        full_fname_img = os.path.join(args.dir_input, fname_img_in)


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
        img_res_vis_temp2 = process_img_my(full_fname_img, proc_size, device, model, loader, param_name_model)


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


        ###------------------------------------------------------------------------------------------------
        ### routine for (bev)
        ###------------------------------------------------------------------------------------------------
        if 0:
            ###------------------------------------------------------------------------------------
            ### create img
            ###------------------------------------------------------------------------------------
            img_raw_in_bev                  = cv2.warpPerspective(img_raw_in,                      tf_ipm, (param_w_img_bev, param_h_img_bev))
            img_res_seg_bev                 = cv2.warpPerspective(img_res_seg,                     tf_ipm, (param_w_img_bev, param_h_img_bev))
            img_res_centerness_direct_bev   = cv2.warpPerspective(img_res_centerness_direct_rgb,   tf_ipm, (param_w_img_bev, param_h_img_bev))
            img_res_centerness_combined_bev = cv2.warpPerspective(img_res_centerness_combined_rgb, tf_ipm, (param_w_img_bev, param_h_img_bev))
            img_res_vis_temp2_bev           = cv2.warpPerspective(img_res_vis_temp2,               tf_ipm, (param_w_img_bev, param_h_img_bev))


            ###------------------------------------------------------------------------------------
            ### show & save
            ###------------------------------------------------------------------------------------

            ###
            cv2.imshow('img_raw_in_bev', img_raw_in_bev)
            cv2.imwrite(fname_out_img_raw_in_bev, img_raw_in_bev)
            cv2.waitKey(1)

            ###
            # cv2.imshow('img_res_seg_bev', img_res_seg_bev)
            # cv2.imwrite(fname_out_img_res_seg_bev, img_res_seg_bev)
            # cv2.waitKey(1)

            ###
            img_res_seg_bev2 = cv2.addWeighted(src1=img_raw_in_bev, alpha=0.5, src2=img_res_seg_bev, beta=0.5, gamma=0)
            cv2.imshow('img_res_seg_bev2', img_res_seg_bev2)
            cv2.imwrite(fname_out_img_res_seg_bev, img_res_seg_bev2)
            cv2.waitKey(1)

            ###
            img_centerness_direct_rgb_bev2 = cv2.addWeighted(src1=img_raw_in_bev, alpha=0.3, src2=img_res_centerness_direct_bev, beta=0.7, gamma=0)
            cv2.imshow('img_centerness_direct_rgb_bev2', img_centerness_direct_rgb_bev2)
            cv2.imwrite(fname_out_img_res_centerness_direct_bev, img_centerness_direct_rgb_bev2)
            cv2.waitKey(1)

            ###
            img_centerness_combined_rgb_bev2 = cv2.addWeighted(src1=img_raw_in_bev, alpha=0.3, src2=img_res_centerness_combined_bev, beta=0.7, gamma=0)
            cv2.imshow('img_centerness_combined_rgb_bev2', img_centerness_combined_rgb_bev2)
            cv2.imwrite(fname_out_img_res_centerness_combined_bev, img_centerness_combined_rgb_bev2)
            cv2.waitKey(1)


            ###
            img_res_vis_temp2_bev2 = cv2.addWeighted(src1=img_raw_in_bev, alpha=0.3, src2=img_res_vis_temp2_bev, beta=0.7, gamma=0)
            cv2.imshow('img_res_vis_temp2_bev2', img_res_vis_temp2_bev2)
            cv2.imwrite(fname_out_img_res_vis_temp2_bev, img_res_vis_temp2_bev2)
            cv2.waitKey(1)
        #end



    #end
#end


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

