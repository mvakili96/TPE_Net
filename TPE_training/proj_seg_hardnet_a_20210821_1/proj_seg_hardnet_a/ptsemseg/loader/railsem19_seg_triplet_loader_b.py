# 2020/07/20
# Written by Jungwon

# <2020/7/10>
#   check rgb, bgr !!!
#   check resizing error: please, check white pixels when resizing in seg-label


# <to-do>
#   - making unlabelled pixels to 250
#   - resizing
#   - considering mean offset
#------------------------------------------------------------------------------------------------------------------
# [Note] (by Jungwon)
#   RailSem19_SegTriplet_b_Loader()
#       __init__()
#       __len__()           : returns the total number of imgs (e.g. total OOOO training imgs)
#       __getitem__()
#------------------------------------------------------------------------------------------------------------------


import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

from torch.utils import data
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate


from ptsemseg.loader.myhelpers import myhelper_railsem19_b

#=======================================================================================================================
# This loader uses the following two datasets:
# 1) semantic segmentation
#   <root>              /home/yu1/proj_avin/dataset/rs19_val
#   <info>              /home/yu1/proj_avin/dataset/rs19_val/rs19-config.json
#   <raw img>           /home/yu1/proj_avin/dataset/rs19_val/jpgs/rs19_val/rsXXXXX.jpg : rgb image (ch3)
#   <pixelwise labels>  /home/yu1/proj_avin/dataset/rs19_val/uint8/rs19_val/rsXXXXX.png: label image (ch1)
#
# 2) triplet
#   <root> /home/yu1/proj_avin/dataset/rs19_triplet
#   <data> /home/yu1/proj_avin/dataset/rs19_triplet/my_triplet_json/rsXXXXX.txt
#=======================================================================================================================


########################################################################################################################
### class RailSem19_SegTriplet_Loader
########################################################################################################################
class RailSem19_SegTriplet_b_Loader(data.Dataset):
    ###############################################################################################################
    ### RailSem19_SegTriplet_Loader::__init__()
    ###############################################################################################################
    def __init__(self,
                 dir_root_data_seg="",
                 dir_root_data_triplet="",
                 type_trainval="train",
                 b_do_transform=False,
                 augmentations=None,
                 output_size_hmap="size_fmap",
                 n_classes_seg = 19,
                 n_channels_reg = 3,
                 network_input_size = None,
                 arch_this = None):

        # output_size_hmap : "size_img_rsz" or "size_fmap"


        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        ### set
        ###///////////////////////////////////////////////////////////////////////////////////////////////////

        ###=============================================================================================
        ### 1. set from external
        ###=============================================================================================
        self.dir_root_data_seg      = dir_root_data_seg
        self.dir_root_data_triplet  = dir_root_data_triplet

        self.type_trainval          = type_trainval
        self.b_do_transform         = b_do_transform
        self.augmentations          = augmentations
        self.output_size_hmap       = output_size_hmap


        ###=============================================================================================
        ### 2. set in internal
        ###=============================================================================================
        self.rgb_mean               = np.array([128.0, 128.0, 128.0])/255.0     # for pixel value 0.0 ~ 1.0
        self.rgb_std                = np.array([1.0, 1.0, 1.0])                 # for pixel value 0.0 ~ 1.0
        self.n_classes              = n_classes_seg
        self.n_channels             = n_channels_reg

        self.size_img_ori           = {'h': 1080, 'w': 1920}    # FIXED, DO NOT EDIT
        self.size_img_rsz           = network_input_size

        self.down_ratio_rsz_fmap    = 4                         # img_rsz/fmap
        self.size_fmap              = {'h': (540 // self.down_ratio_rsz_fmap),
                                       'w': (960 // self.down_ratio_rsz_fmap)}


        self.arch_this = arch_this
        ###
        self._set_FACTOR()


        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        ### read fnames
        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        self.dir_img_raw_jpg    = dir_root_data_seg     + 'jpgs/rs19_val/'
        if self.n_classes == 19:
            self.dir_label_seg_png  = dir_root_data_seg     + 'uint8/rs19_val/'
        elif self.n_classes == 3:
            self.dir_label_seg_png = dir_root_data_seg + 'uint8/rs19_val_modified/'

        self.dir_label_ins_json = dir_root_data_seg     + 'uint8/instance_json/'


        self.dir_triplet_json   = dir_root_data_triplet + 'my_triplet_json/'

        if self.n_channels == 1:
            self.dir_triplet_image   = dir_root_data_triplet + 'my_triplet_image/'
        elif self.n_channels == 3:
            self.dir_triplet_C = dir_root_data_triplet + 'my_triplet_C/'
            self.dir_triplet_L = dir_root_data_triplet + 'my_triplet_L/'
            self.dir_triplet_R = dir_root_data_triplet + 'my_triplet_R/'

        self.dir_points_poly_regression = 'TARGET/'


        ###=============================================================================================
        ### 3. read fnames for all the raw-imgs
        ###=============================================================================================
        self.fnames_img_raw_jpg = myhelper_railsem19_b.read_fnames_trainval(self.dir_img_raw_jpg, 6000)


        ###=============================================================================================
        ### 4. read fnames for all the seg-labels
        ###=============================================================================================
        self.fnames_label_seg_png = myhelper_railsem19_b.read_fnames_trainval(self.dir_label_seg_png, 6000)




        ###=============================================================================================
        ### 5. read fname for all the triplets (json)
        ###=============================================================================================
        self.fnames_triplet_json = myhelper_railsem19_b.read_fnames_trainval(self.dir_triplet_json, 6000)


        ###=============================================================================================
        ### 6. read fname for all the instances
        ###=============================================================================================
        self.fnames_ins_json = myhelper_railsem19_b.read_fnames_trainval(self.dir_label_ins_json, 6000)

        ###=============================================================================================
        ### 7. read fname for all the triplets (image)
        ###=============================================================================================
        if self.n_channels == 1:
            self.fnames_triplet_image = myhelper_railsem19_b.read_fnames_trainval(self.dir_triplet_image, 6000)
        elif self.n_channels == 3:
            self.fnames_triplet_C = myhelper_railsem19_b.read_fnames_trainval(self.dir_triplet_C, 6000)
            self.fnames_triplet_L = myhelper_railsem19_b.read_fnames_trainval(self.dir_triplet_L, 6000)
            self.fnames_triplet_R = myhelper_railsem19_b.read_fnames_trainval(self.dir_triplet_R, 6000)

        ###=============================================================================================
        ### 8. read fname for all the triplets (json)
        ###=============================================================================================
        self.fnames_points_poly_regression = myhelper_railsem19_b.read_fnames_trainval(self.dir_points_poly_regression, 6000)


    #end



    ###############################################################################################################
    ### RailSem19_SegTriplet_Loader::_set_FACTOR()
    ###############################################################################################################
    def _set_FACTOR(self):
        ###=============================================================================================
        ### automatically-set
        ###=============================================================================================
        self.FACTOR_ori_to_rsz_h = float(self.size_img_rsz['h'])/float(self.size_img_ori['h'])
        self.FACTOR_ori_to_rsz_w = float(self.size_img_rsz['w'])/float(self.size_img_ori['w'])

        self.FACTOR_ori_to_fmap_h = float(self.size_fmap['h'])/float(self.size_img_ori['h'])
        self.FACTOR_ori_to_fmap_w = float(self.size_fmap['w'])/float(self.size_img_ori['w'])

        self.FACTOR_rsz_to_fmap_h = float(self.size_fmap['h'])/float(self.size_img_rsz['h'])
        self.FACTOR_rsz_to_fmap_w = float(self.size_fmap['w'])/float(self.size_img_rsz['w'])
    #end


    ###############################################################################################################
    ### RailSem19_SegTriplet_Loader::__len__()
    ###############################################################################################################
    def __len__(self):
        return len(self.fnames_img_raw_jpg[self.type_trainval])
        # return the total number of images belong to self.type_trainval
    #end


    ###############################################################################################################
    ### RailSem19_SegTriplet_Loader::__getitem__()
    ###############################################################################################################
    def __getitem__(self, index):
        # << read ONE image & seg-label & triplet >>

        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        ### 1. read from files
        ###///////////////////////////////////////////////////////////////////////////////////////////////////

        ###=============================================================================================
        ### 1.1 set full-fname
        ###=============================================================================================
        full_fname_img_raw_jpg    = self.dir_img_raw_jpg    + self.fnames_img_raw_jpg  [self.type_trainval][index]
        full_fnames_label_seg_png = self.dir_label_seg_png  + self.fnames_label_seg_png[self.type_trainval][index]
        full_fnames_ins_json      = self.dir_label_ins_json + self.fnames_ins_json[self.type_trainval][index]
        full_fname_triplet_json   = self.dir_triplet_json   + self.fnames_triplet_json [self.type_trainval][index]
        if self.n_channels == 1:
            full_fname_triplet_image  = self.dir_triplet_image  + self.fnames_triplet_image [self.type_trainval][index]
        elif self.n_channels == 3:
            full_fname_triplet_C  = self.dir_triplet_C  + self.fnames_triplet_C [self.type_trainval][index]
            full_fname_triplet_L  = self.dir_triplet_L  + self.fnames_triplet_L [self.type_trainval][index]
            full_fname_triplet_R  = self.dir_triplet_R  + self.fnames_triplet_R [self.type_trainval][index]

        full_fname_points_poly_regression = self.dir_points_poly_regression + self.fnames_points_poly_regression[self.type_trainval][index]




        ###=============================================================================================
        ### 1.2 read img_raw_jpg (from file)
        ###=============================================================================================
        img_raw_rsz_uint8, \
        img_raw_rsz_fl_n = myhelper_railsem19_b.read_img_raw_jpg_from_file(full_fname_img_raw_jpg,
                                                                           self.size_img_rsz,
                                                                           self.arch_this,
                                                                           self.rgb_mean,
                                                                           self.rgb_std)

        ###=============================================================================================
        ### 1.3 read label_seg_png (from file)
        ###=============================================================================================
        img_label_seg_rsz_uint8 = myhelper_railsem19_b.read_label_seg_png_from_file(full_fnames_label_seg_png,
                                                                                    self.size_img_rsz)


        ###=============================================================================================
        ### 1.4 read list_triplet_json (from file)
        ###=============================================================================================
        list_triplet_json = json.load(open(full_fname_triplet_json, 'r'))


        ###=============================================================================================
        ### 1.5 read list_ins_json (from file)
        ###=============================================================================================
        list_ins_json = json.load(open(full_fnames_ins_json, 'r'))
        list_ins_json = np.array(list_ins_json)

        ###=============================================================================================
        ### 1.6 read triplet image (from file)
        ###=============================================================================================
        if self.n_channels == 1:
            labelmap_centerline = myhelper_railsem19_b.read_triplet_image_from_file(full_fname_triplet_image,self.size_img_rsz)
        elif self.n_channels == 3:
            labelmap_centerline = myhelper_railsem19_b.read_triplet_C_from_file(full_fname_triplet_C,self.size_img_rsz)
            labelmap_leftrail   = myhelper_railsem19_b.read_triplet_image_from_file(full_fname_triplet_L,self.size_img_rsz)
            labelmap_rightrail  = myhelper_railsem19_b.read_triplet_image_from_file(full_fname_triplet_R,self.size_img_rsz)
            labelmap_leftright = np.concatenate((labelmap_leftrail, labelmap_rightrail), axis=0)
            output_labelmap_leftright  = torch.from_numpy(labelmap_leftright).float()

        ###=============================================================================================
        ### 1.7 read list_ins_json (from file)
        ###=============================================================================================
        list_polypoints_json = json.load(open(full_fname_points_poly_regression, 'r'))
        list_polypoints      = np.array(list_polypoints_json)


        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        ### 2. processing
        ###///////////////////////////////////////////////////////////////////////////////////////////////////

        ###=============================================================================================
        ### 2.1 post-processing label_seg_png
        ###=============================================================================================
        # note that labels other than 0~18 should be 250 (which indicates invalid)

        set_idx_invalid = (img_label_seg_rsz_uint8 > 18)
            # completed to set
            #       set_idx_invalid: (h, w), bool

        img_label_seg_rsz_uint8[set_idx_invalid] = 250
            # completed to set
            #       img_label_seg_rsz_uint8: (h, w), uint8



        ###=============================================================================================
        ### 2.2 create hmap from list_triplet_json
        ###=============================================================================================
        # labelmap_centerline = None
        # labelmap_leftrail = None
        # labelmap_rightrail = None

        # if (self.output_size_hmap) is "size_fmap":
        #     labelmap_centerline, \
        #     labelmap_leftrail,\
        #     labelmap_rightrail = myhelper_railsem19_b.create_labelmap_triplet(list_triplet_json,
        #                                                                       self.size_fmap,
        #                                                                       self.FACTOR_ori_to_fmap_h,
        #                                                                       self.FACTOR_ori_to_fmap_w)
        #
        #     labelmap_centerline_priority = myhelper_railsem19_b.create_labelmap_triplet_priority(list_triplet_json,
        #                                                                       self.size_fmap,
        #                                                                       self.FACTOR_ori_to_fmap_h,
        #                                                                       self.FACTOR_ori_to_fmap_w)
        # elif (self.output_size_hmap) is "size_img_rsz":
        #     labelmap_centerline, \
        #     labelmap_leftrail, \
        #     labelmap_rightrail = myhelper_railsem19_b.create_labelmap_triplet(list_triplet_json,
        #                                                                       self.size_img_rsz,
        #                                                                       self.FACTOR_ori_to_rsz_h,
        #                                                                       self.FACTOR_ori_to_rsz_w)
        #
        #     labelmap_centerline_priority = myhelper_railsem19_b.create_labelmap_triplet_priority(list_triplet_json,
        #                                                                       self.size_img_rsz,
        #                                                                       self.FACTOR_ori_to_rsz_h,
        #                                                                       self.FACTOR_ori_to_rsz_w)

        # labelmap_leftright = np.concatenate((labelmap_leftrail, labelmap_rightrail), axis=0)




        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        ### 3. output
        ###///////////////////////////////////////////////////////////////////////////////////////////////////

        ###
        output_img_raw             = torch.from_numpy(img_raw_rsz_fl_n).float()
        output_img_label_seg       = torch.from_numpy(img_label_seg_rsz_uint8).long()
        output_ins_json            = torch.from_numpy(list_ins_json)
        output_polypoints          = torch.from_numpy(list_polypoints)
        output_labelmap_centerline = torch.from_numpy(labelmap_centerline).float()
        # output_labelmap_centerline_priority = torch.from_numpy(labelmap_centerline_priority).float()

        ###
        if self.n_channels == 1:
            output_final = {'img_raw_fl_n'                   : output_img_raw,                              # (3, h_rsz, w_rsz)
                            'gt_img_label_seg'               : output_img_label_seg,                        # (h_rsz, w_rsz)
                            'gt_instances'                   : output_ins_json,                             # (h_rsz, w_rsz)
                            'gt_labelmap_centerline'         : output_labelmap_centerline,
                            'gt_polypoints'                  : output_polypoints}

        elif self.n_channels == 3:
            output_final = {'img_raw_fl_n'                   : output_img_raw,                              # (3, h_rsz, w_rsz)
                            'gt_img_label_seg'               : output_img_label_seg,                        # (h_rsz, w_rsz)
                            'gt_instances'                   : output_ins_json,                             # (h_rsz, w_rsz)
                            'gt_labelmap_centerline'         : output_labelmap_centerline,                 # (1, h_rsz, w_rsz)
                            'gt_labelmap_leftright'          : output_labelmap_leftright,
                            'gt_polypoints'                  : output_polypoints}


        return output_final
    #end
#end




########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
### __main__
########################################################################################################################
if __name__ == "__main__":
    ###============================================================================================================
    ### setting
    ###============================================================================================================
    batch_size = 1


    ###============================================================================================================
    ### create objects for dataloader
    ###============================================================================================================

    ### (1) create an object1 for dataloader
    trainloader_head = RailSem19_SegTriplet_b_Loader(output_size_hmap="size_img_rsz")
        # completed to create
        #       trainloader_head


    ### (2) create an object2 for dataloader
    trainloader_batch = data.DataLoader(trainloader_head, batch_size=batch_size)
        # completed to create
        #       trainloader


    ###============================================================================================================
    ### loop
    ###============================================================================================================

    ### (1) create fig
    fig_plt, axarr = plt.subplots(batch_size, 2)
        # completed to set
        #       fig_plt: fig object
        #       axarr:   axes object

    ### (2) loop
    for idx_this, data_samples in enumerate(trainloader_batch):
        # i: 0 ~ 91
        #   note that there are OOOO training images, which means that there are idx: 0 ~ OOOO for training imgs
        #       idx_loop -> (batch_size*idx_loop) ~ (batch_size*(idx_loop+1) - 1)
        #       idx_loop:0 -> 0 ~ 3     (if batch_size: 4)
        #       idx_loop:1 -> 4 ~ 7
        # data_samples: list from trainloader
        #       [0]: imgs with size batch_size
        #       [1]: labels with size batch_size
        print('showing {}'.format(idx_this))


        ###------------------------------------------------------------------------------------------
        ### get batch_data
        ###------------------------------------------------------------------------------------------
        batch_img_raw       = data_samples['img_raw_fl_n']
        batch_img_label_seg = data_samples['gt_img_label_seg']
        batch_hmap          = data_samples['gt_labelmap_centerline']


        ###------------------------------------------------------------------------------------------
        ### conversion
        ###------------------------------------------------------------------------------------------
        batch_img_raw = batch_img_raw.numpy()[:, ::-1, :, :]        # BGR -> RGB (for using axarr[][].imshow()
            # batch_img_raw: (bs, ch, h, w), RGB


        if 0:
            ### show
            for idx_bs in range(batch_size):

                ### img_raw
                img_raw = batch_img_raw[idx_bs]
                img_raw_vis = myhelper_railsem19_b.convert_img_data_to_img_ori(img_raw)
                #img_raw_vis = trainloader_head.convert_img(img_raw)

                ### img_label_seg
                img_label_seg = batch_img_label_seg.numpy()[idx_bs]
                    # img_label_seg: ndarray (360, 480), val 0~18, 250

                img_label_seg_decoded = myhelper_railsem19_b.decode_segmap(img_label_seg)
                    # img_label_seg_decoded: ndarray (360, 480, 3)

                ### show
                if batch_size >= 2:
                    axarr[idx_bs][0].imshow(img_raw_vis)                    # show image
                    axarr[idx_bs][1].imshow(img_label_seg_decoded)          # show labelmap
                else:
                    axarr[0].imshow(img_raw_vis)                            # show image
                    axarr[1].imshow(img_label_seg_decoded)                  # show labelmap
                #end
            #end

            #plt.show()
            str_title = '%d' % idx_this
            fig_plt.suptitle(str_title)
            plt.draw()
            plt.pause(1)
        #end


        #if idx_this >= 10:
        #    break
        #end
    #end


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################






