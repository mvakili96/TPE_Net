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
from ptsemseg.loader.myhelpers  import myhelper_railsem19
from helpers_my                 import my_utils

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


    ###
    img_raw_rsz_uint8, \
    img_raw_rsz_fl_n = myhelper_railsem19.read_img_raw_jpg_from_file(full_fname_img_raw_jpg, loader.size_img_rsz)


    img_raw = np.expand_dims(img_raw_rsz_fl_n, 0)
    img_raw = torch.from_numpy(img_raw).float()

    images = img_raw.to(device)


    ###
    outputs_seg = None
    outputs_hmap_centerline = None

    if param_name_model["arch"] is "hardnet":
        outputs_seg = model(images)
    elif param_name_model["arch"] is "rpnet_b":
        outputs_seg, outputs_hmap_centerline = model(images)
    #end


    ###
    labels_predicted = np.squeeze(outputs_seg.data.max(1)[1].cpu().numpy(), axis=0)
    img_res_seg = myhelper_railsem19.decode_segmap_b(labels_predicted)


    ###
    hmap_sigmoid = torch.sigmoid(outputs_hmap_centerline)
    hmap_out_a = hmap_sigmoid[0].permute(1, 2, 0)   # hmap_out_a : tensor(512, 1024, 1)
    hmap_out_b = hmap_out_a[:, :, 0]                # hmap_out_b : tensor(512, 1024)
    hmap_out_c = hmap_out_b * 255.0
    hmap_out_d = torch.clamp(hmap_out_c, min=0.0, max=255.0)

    hmap_out_e = hmap_out_d.detach().cpu().numpy()
    img_hmap_out = hmap_out_e.astype(np.uint8)
        # completed to set
        #       img_hmap_out: ndarray (h,w), uint8


    return img_raw_rsz_uint8, img_res_seg, img_hmap_out
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
    #param_file_weight = "/home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/rpnet_b_railsem19_seg_triplet0/cur/rpnet_b_railsem19_seg_triplet_best_model_22900_new.pkl"
    param_file_weight = "/home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/rpnet_b_railsem19_seg_triplet0/cur/rpnet_b_railsem19_seg_triplet_best_model_69500.pkl"
    #param_file_weight = "/home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/rpnet_b_railsem19_seg_triplet0/cur_20200714_1/rpnet_b_railsem19_seg_triplet_best_model_59200.pkl"
    #param_file_weight = "/home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/rpnet_railsem19_seg_triplet0/cur/rpnet_a_railsem19_seg_triplet_best_model.pkl"
    #param_file_weight = "/home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/hardnet_railsem19_my2_20200502_1/cur/hardnet_railsem19_best_model.pkl"
    #param_name_model  = {"arch": "hardnet"}
    param_name_model = {"arch": "rpnet_b"}
        # please, see <ptsemseg.models.__init__.py>
    param_dataset     = "railsem19_seg_triplet"
        # please, see <ptsemseg.loader.__init__.py>
    param_size        = "540,960"
    param_dir_input   = "/media/yu1/hdd_my/Dataset_YDHR_OCT009_OCT10/img_gopro/selected/test7_run2_normal_reverse_switch"
    param_dir_output  = "/home/yu1/Desktop/temp3a"


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
    ### (4) loop
    ###============================================================================================================
    print("Process all image inside : {}".format(args.dir_input))

    list_fnames_img_ = os.listdir(args.dir_input)
    list_fnames_img  = sorted(list_fnames_img_)


    for fname_img in list_fnames_img:
        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        _, ext = os.path.splitext(os.path.basename((fname_img)))
        if ext not in [".png", ".jpg"]:
            continue
        #end


        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        full_fname_img = os.path.join(args.dir_input, fname_img)


        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        img_raw_in, \
        img_res_seg, \
        img_res_hmap = process_img_my(full_fname_img, proc_size, device, model, loader, param_name_model)


        ###
        fname_out_img = os.path.join(args.dir_output, os.path.basename(fname_img))
        cv2.imwrite(fname_out_img, img_res_seg)
        #misc.imsave(out_path, decoded)


        if 1:
            ### temp routine
            # set_idx_pos_temp = img_res_hmap >= 128
            # set_idx_neg_temp = img_res_hmap < 128
            #
            # img_res_hmap[set_idx_pos_temp] = 255
            # img_res_hmap[set_idx_neg_temp] = 0


            img_hmap_rgb = cv2.cvtColor(img_res_hmap, cv2.COLOR_GRAY2BGR)

            #img_hmap_out2 = cv2.addWeighted(src1=img_raw_in, alpha=0.25, src2=img_hmap_rgb, beta=0.75, gamma=0)
            #cv2.imshow('img_hmap_out2', img_hmap_out2)
            cv2.imshow('img_hmap_out2', img_hmap_rgb)
            cv2.imwrite(fname_out_img, img_hmap_rgb)
            cv2.waitKey(1)
        # end


    #end
#end


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

#   --model_path /home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/hardnet_railsem19_my2_20200502_1/cur/hardnet_railsem19_best_model.pkl  --dataset railsem19  --size 540,960  --input /media/yu1/hdd_my/Dataset_YDHR_OCT009_OCT10/img_gopro/selected/test7_run2_normal_reverse_switch  --output /home/yu1/Desktop/temp3





"""
def process_img_my(img_path, size, device, model, loader):
    print("Read Input Image from : {}".format(img_path))

    rgb_mean = np.array([128.0, 128.0, 128.0])

    img = misc.imread(img_path)
    img = misc.imresize(img, (loader.size_img_rsz['h'], loader.size_img_rsz['w']))
    img_resized = copy.deepcopy(img)

    img = img[:, :, ::-1]       # rgb -> bgr
    img = img.astype(np.float64)
    img -= rgb_mean

    if 1:
        img = img.astype(float) / 255.0
    #end


    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

    decoded = myhelper_railsem19.decode_segmap(pred)
    #decoded = loader.decode_segmap(pred)

    return img_resized, decoded
#end
"""