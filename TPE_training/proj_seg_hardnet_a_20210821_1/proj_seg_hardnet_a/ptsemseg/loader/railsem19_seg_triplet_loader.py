# 2020/07/10
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
#   RailSem19Loader()
#       __init__()
#       __len__()           : returns the total number of imgs (e.g. total OOOO training imgs)
#       __getitem__()
#           transform()     : only called in __getitem__()
#       decode_segmap()     : only called in  __main__()
#
# *installed imageio for imread
#------------------------------------------------------------------------------------------------------------------


import os
import collections
import torch
import cv2
import json
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import copy

from torch.utils import data
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate

### <newly added>
import imageio                      # for using imageio.imread() instead of m.imread()
from PIL import Image

from ptsemseg.loader.myhelpers import myhelper_railsem19
from ptsemseg.loader.myhelpers.lib.image import draw_umich_gaussian, gaussian_radius

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
class RailSem19_SegTriplet_Loader(data.Dataset):
    ###############################################################################################################
    ### RailSem19_SegTriplet_Loader::__init__()
    ###############################################################################################################
    def __init__(self,
                 dir_root_data_seg="/home/yu1/proj_avin/dataset/rs19_val/",
                 dir_root_data_triplet="/home/yu1/proj_avin/dataset/rs19_triplet/",
                 type_trainval="train",
                 b_do_transform=False,
                 augmentations=None,
                 output_size_hmap="size_fmap"):

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
        self.n_classes              = 19

        self.size_img_ori           = {'h': 1080, 'w': 1920}    # FIXED, DO NOT EDIT
        self.size_img_rsz           = {'h': 540,  'w': 960}
        self.down_ratio_rsz_fmap    = 4                         # img_rsz/fmap
        self.size_fmap              = {'h': (540 // self.down_ratio_rsz_fmap),
                                       'w': (960 // self.down_ratio_rsz_fmap)}

        ###
        self._set_FACTOR()


        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        ### read fnames
        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        self.dir_img_raw_jpg    = dir_root_data_seg     + 'jpgs/rs19_val/'
        self.dir_label_seg_png  = dir_root_data_seg     + 'uint8/rs19_val/'
        self.dir_triplet_json   = dir_root_data_triplet + 'my_triplet_json/'


        ###=============================================================================================
        ### 3. read fnames for all the raw-imgs
        ###=============================================================================================
        self.fnames_img_raw_jpg = myhelper_railsem19.read_fnames_trainval(self.dir_img_raw_jpg, 7000)
            # completed to set
            #       self.fnames_img_raw_jpg["train"]: list for fnames only (for train)
            #       self.fnames_img_raw_jpg["val"]  : list for fnames only (for val)
            #               (e.g. 'rs01001.jpg', 'rs01002.jpg', ...)


        ###=============================================================================================
        ### 4. read fnames for all the seg-labels
        ###=============================================================================================
        self.fnames_label_seg_png = myhelper_railsem19.read_fnames_trainval(self.dir_label_seg_png, 7000)
            # completed to set
            #       self.fnames_label_seg_png["train"]: list for fnames only (for train)
            #       self.fnames_label_seg_png["val"]  : list for fnames only (for val)
            #               (e.g. 'rs01001.png', 'rs01002.png', ...)


        ###=============================================================================================
        ### 5. read fname for all the triplets
        ###=============================================================================================
        self.fnames_triplet_json = myhelper_railsem19.read_fnames_trainval(self.dir_triplet_json, 7000)
            # completed to set
            #       self.fnames_triplet_json["train"]: list for fnames only (for train)
            #       self.fnames_triplet_json["val"]  : list for fnames only (for val)
            #               (e.g. 'rs01001.txt', 'rs01002.txt', ...)
    #end

        #---------------------------------------------------------------------------------------------
        # completed to set
        #   self.dir_root_data_seg
        #   self.dir_root_data_triplet
        #
        #   self.type_trainval
        #   self.b_do_transform
        #   self.augmentations
        #   self.b_resize
        #
        #   self.rgb_mean
        #   self.n_classes
        #   self.size_img_rsz[]
        #
        #   self.dir_img_raw_jpg
        #   self.dir_label_seg_png
        #   self.dir_triplet_json
        #
        #   self.fnames_img_raw_jpg[]
        #   self.fnames_label_seg_png[]
        #   self.fnames_triplet_json[]
        #---------------------------------------------------------------------------------------------
        # <<How to get full-filename>>
        #   self.dir_img_raw_jpg    + self.fnames_img_raw_jpg[i]
        #   self.dir_label_seg_png  + self.fnames_label_seg_png[i]
        #   self.dir_triplet_json   + self.fnames_triplet_json[i]
        #---------------------------------------------------------------------------------------------


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
        ### set full-fname
        ###=============================================================================================
        full_fname_img_raw_jpg    = self.dir_img_raw_jpg   + self.fnames_img_raw_jpg  [self.type_trainval][index]
        full_fnames_label_seg_png = self.dir_label_seg_png + self.fnames_label_seg_png[self.type_trainval][index]
        full_fname_triplet_json   = self.dir_triplet_json  + self.fnames_triplet_json [self.type_trainval][index]
            # completed to set
            #   full_fname_img_raw_jpg
            #   full_fnames_label_seg_png
            #   full_fname_triplet_json


        ###=============================================================================================
        ### read img_raw_jpg (from file)
        ###=============================================================================================
        img_raw_rsz_uint8, \
        img_raw_rsz_fl_n = myhelper_railsem19.read_img_raw_jpg_from_file(full_fname_img_raw_jpg,
                                                                         self.size_img_rsz,
                                                                         self.rgb_mean,
                                                                         self.rgb_std)
            # completed to set
            #       img_raw_rsz_fl_n: -1.0 ~ 1.0, ndarray(C,H,W), float32


        ###=============================================================================================
        ### read label_seg_png (from file)
        ###=============================================================================================
        img_label_seg_rsz_uint8 = myhelper_railsem19.read_label_seg_png_from_file(full_fnames_label_seg_png,
                                                                                  self.size_img_rsz)
            # completed to set
            #       img_label_seg_rsz_uint8: 0 ~ 255, ndarray(H,W), uint8


        ###=============================================================================================
        ### read list_triplet_json (from file)
        ###=============================================================================================

        ### get list_triplet_json
        list_triplet_json = json.load(open(full_fname_triplet_json, 'r'))
            # completed to set
            #       list_triplet_json
            # *see the following file for detailed info about 'list_triplet_json':
            #   </home/yu1/proj_avin/dataset/proj_rs19_jungwon_a/main_create_LRC_triplet_my.py>

        ### <<debugging>>
        if 0:
            img_vis_label_my_triplet = copy.deepcopy(img_raw_rsz_uint8)
            myhelper_railsem19.visualize_label_my_triplet(list_triplet_json,
                                                          img_vis_label_my_triplet,
                                                          self.FACTOR_ori_to_rsz_h,
                                                          self.FACTOR_ori_to_rsz_w)
        #end



        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        ### 2. processing
        ###///////////////////////////////////////////////////////////////////////////////////////////////////

        ###=============================================================================================
        ### post-processing label_seg_png
        ###=============================================================================================
        # note that labels other than 0~18 should be 250 (which indicates invalid)

        set_idx_invalid = (img_label_seg_rsz_uint8 > 18)
            # completed to set
            #       set_idx_invalid: (h, w), bool

        img_label_seg_rsz_uint8[set_idx_invalid] = 250
            # completed to set
            #       img_label_seg_rsz_uint8: (h, w), uint8



        ###=============================================================================================
        ### create hmap from list_triplet_json
        ###=============================================================================================
        # output_size_hmap : "size_img_rsz" or "size_fmap"
        hmap_centerline = None

        if (self.output_size_hmap) is "size_fmap":
            hmap_centerline = myhelper_railsem19.create_hmap_fmap(list_triplet_json,
                                                                   self.size_fmap,
                                                                   self.FACTOR_ori_to_fmap_h,
                                                                   self.FACTOR_ori_to_fmap_w)
        elif (self.output_size_hmap) is "size_img_rsz":
            hmap_centerline = myhelper_railsem19.create_hmap_fmap(list_triplet_json,
                                                                   self.size_img_rsz,
                                                                   self.FACTOR_ori_to_rsz_h,
                                                                   self.FACTOR_ori_to_rsz_w)
        #end
            # completed to set
            #       hmap_centerline: ndarray(num_class, h, w), here, num_class = 1


        ### <<debugging>>
        if 0:
            #myhelper_railsem19.visualize_hmap(hmap_centerline[0])
            img_hmap_rgb = myhelper_railsem19.visualize_hmap_b(hmap_centerline[0], img_raw_rsz_uint8)

            ###
            fname_common = "/home/yu1/Desktop/dir_temp/temp4a/"

            fname_img_hmap_rgb      = fname_common + 'hmap_' + str(index) + '.jpg'
            fname_img_raw_rsz_uint8 = fname_common + 'in_'   + str(index) + '.jpg'

            cv2.imwrite(fname_img_hmap_rgb,      img_hmap_rgb)
            cv2.imwrite(fname_img_raw_rsz_uint8, img_raw_rsz_uint8)
        #end



        ###///////////////////////////////////////////////////////////////////////////////////////////////////
        ### 3. output
        ###///////////////////////////////////////////////////////////////////////////////////////////////////

        ###
        output_img_raw          = torch.from_numpy(img_raw_rsz_fl_n).float()
        output_img_label_seg    = torch.from_numpy(img_label_seg_rsz_uint8).long()
        output_hmap_centerline  = torch.from_numpy(hmap_centerline).float()


        ###
        output_final = {'img_raw_fl_n': output_img_raw,                 # (3, h_rsz, w_rsz)
                        'gt_img_label_seg': output_img_label_seg,       # (h_rsz, w_rsz)
                        'gt_hmap_centerline': output_hmap_centerline}   # (1, h_fmap, w_fmap)

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
### __main__ (old)
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
    trainloader_head = RailSem19_SegTriplet_Loader(output_size_hmap="size_img_rsz")
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
        batch_hmap          = data_samples['gt_hmap_centerline']


        ###------------------------------------------------------------------------------------------
        ### conversion
        ###------------------------------------------------------------------------------------------
        batch_img_raw = batch_img_raw.numpy()[:, ::-1, :, :]        # BGR -> RGB (for using axarr[][].imshow()
            # batch_img_raw: (bs, ch, h, w), RGB


        if 1:
            ### show
            for idx_bs in range(batch_size):

                ### img_raw
                img_raw = batch_img_raw[idx_bs]
                img_raw_vis = myhelper_railsem19.convert_img_data_to_img_ori(img_raw)
                #img_raw_vis = trainloader_head.convert_img(img_raw)

                ### img_label_seg
                img_label_seg = batch_img_label_seg.numpy()[idx_bs]
                    # img_label_seg: ndarray (360, 480), val 0~18, 250

                img_label_seg_decoded = myhelper_railsem19.decode_segmap(img_label_seg)
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


"""
########################################################################################################################
### __main__ (old)
########################################################################################################################
if __name__ == "__main__":
    ###============================================================================================================
    ### setting
    ###============================================================================================================

    ### (1) set path for dataset
    #path_root = "/media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417"
    #path_root = "/home/yu1/proj_avin/dataset/rs19_val"
        # completed to set
        #       path_root
        #---------------------------------------------------------------------------------------------
        # note that there are the following folders:
        #   /media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417
        #       : dir_root_data_seg
        #   /media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417/jpgs/rs19_val
        #       : rgb image (ch3), jpg
        #   /media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417/uint8/rs19_val
        #       : label image (ch1), png
        #---------------------------------------------------------------------------------------------

    ### (2) create an object for augmentations
    augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip(0.5)])
        # completed to create
        #       augmentations

    ### (3) set batch-size for data loading
    batch_size = 2      # batch_size: batch_size
        # completed to set
        #       batch_size


    ###============================================================================================================
    ### create objects for dataloader
    ###============================================================================================================

    ### (1) create an object1 for dataloader
    #trainloader_head = RailSem19Loader(path_root, b_do_transform=True, augmentations=augmentations)
    trainloader_head = RailSem19_SegTriplet_Loader(b_do_transform=True)
        # completed to create
        #       trainloader_head


    ### (2) create an object2 for dataloader
    trainloader = data.DataLoader(trainloader_head, batch_size=batch_size)
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
    for idx_this, data_samples in enumerate(trainloader):
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
        ### load images & labelmaps
        ###------------------------------------------------------------------------------------------
        images, labelmaps = data_samples


        ###------------------------------------------------------------------------------------------
        ### conversion
        ###------------------------------------------------------------------------------------------
        images = images.numpy()[:, ::-1, :, :]
            # images: ndarray, (bs, 3, 360, 480)

        images = np.transpose(images, [0, 2, 3, 1])
            # images: ndarray, (bs, 360, 480, 3)


        if 1:
            ### show
            for j in range(batch_size):

                ### labelmap
                labelmap_np  = labelmaps.numpy()[j]                             # labelmap_np: ndarray (360, 480), val 0~18, 250
                img_labelmap = myhelper_railsem19.decode_segmap(labelmap_np)
                #img_labelmap = trainloader_head.decode_segmap(labelmap_np)      # img_labelmap: ndarray (360, 480, 3)

                ### show
                axarr[j][0].imshow(images[j])       # show image
                axarr[j][1].imshow(img_labelmap)    # show labelmap
            #end

            #plt.show()
            str_title = '%d' % idx_this
            fig_plt.suptitle(str_title)
            plt.draw()
            plt.pause(1)
        #end

        if idx_this >= 3:
            break
        #end
    #end
"""

"""
###############################################################################################################
### RailSem19_SegTriplet_Loader::_set_FACTOR()
###############################################################################################################
def convert_img(self, img_in):
    # img_in: float32, rgb_mean, rgb_std applied

    img = copy.deepcopy(img_in)
    img = img * self.rgb_std
    img = img + self.rgb_mean

    img = img * 255.0
    img = img.astype(np.uint8)

    return img
# end
"""

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################





