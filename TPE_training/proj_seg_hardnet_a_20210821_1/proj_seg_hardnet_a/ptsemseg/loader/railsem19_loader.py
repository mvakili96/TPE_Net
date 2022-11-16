import os
import collections
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import copy

from torch.utils import data
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate

### <newly added>
import imageio                      # for using imageio.imread() instead of m.imread()
from PIL import Image

# 2020/05/01
# Written by Jungwon

# <to-do>
#   - making unlabelled pixels to 250
#   - resizing
#   - considering mean offset


class RailSem19Loader(data.Dataset):
    #==================================================================================================================
    def __init__(self, fpath_root, type_trainval="train", b_do_transform=False, augmentations=None, b_resize=True):

        ### set from external
        self.fpath_root     = fpath_root
        self.type_trainval  = type_trainval
        self.b_do_transform = b_do_transform
        self.augmentations  = augmentations
        self.b_resize       = b_resize


        ### set in internal
        self.rgb_mean     = np.array([128.0, 128.0, 128.0])
        self.n_classes    = 19
        self.size_img_rsz = [540, 960]        # only meaningful when self.b_resize is True
            # note that original img size: (1080, 1920)


        ### read all the raw-img files of railsem19
        list_fname_img_raw_ = os.listdir(fpath_root + '/jpgs/rs19_val')
        list_fname_img_raw  = sorted(list_fname_img_raw_)
            # completed to set
            #       list_fname_img_raw: list for containing img_names only


        ### set files
        self.fnames = collections.defaultdict(list)
        self.fnames["train"] = list_fname_img_raw[0:7000]
        self.fnames["val"]   = list_fname_img_raw[7000:]
            # completed to set
            #   self.fnames["train"]: list for containing img-names only (for train)
            #   self.fnames["val"]: list for containing img-names only (for train)
            #                 (e.g. 'rs01001.jpg', 'rs01002.jpg', ...)


        # completed to set
        #   [set from external]
        #     self.fpath_root         : root path for dataset
        #     self.type_trainval      : dataset type
        #     self.b_do_transform     : on/off for doing transform
        #     self.augmentations      : on/off for doing augmentations
        #     self.b_resize
        #
        #   [set in internal]
        #     self.n_classes
        #     self.size_img_rsz
        #     self.fnames[]        : list for containing img-names only


    #==================================================================================================================
    def __len__(self):
        return len(self.fnames[self.type_trainval])
        # return the total number of images belong to self.type_trainval
    #==================================================================================================================
    def __getitem__(self, index):
        ###===================================================================================================
        ### read ONE image & labeling
        ###===================================================================================================

        # [note that]
        #   /media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417                   : fpath_root_dataset
        #   /media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417/jpgs/rs19_val     : rgb image (ch3), jpg
        #   /media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417/uint8/rs19_val    : label image (ch1), png


        ###------------------------------------------------------------------------------------------
        ### read img & corresponding labeling
        ###------------------------------------------------------------------------------------------

        ### set fname & path
        fname_image  = self.fnames[self.type_trainval][index]     # fname_image : rs01001.jpg
        fname_image_ = fname_image.split(".")[0]                  # fname_image_: rs01001

        fname_full_image    = self.fpath_root + "/" + "jpgs/rs19_val" + "/" + fname_image
        fname_full_labelmap = self.fpath_root + "/" + "uint8/rs19_val" + "/" + fname_image_ + ".png"
            # completed to set
            #       fname_full_image
            #       fname_full_labelmap


        ### read rgb img
        image0 = imageio.imread(fname_full_image)
        image0 = np.array(image0, dtype=np.uint8)
            # completed to set
            #   image0: ndarray (360, 480, 3), uint8, val 0~255

            # Note that imageio uses RGB. That is:
            #   image0[:,:,0] -> R
            #   image0[:,:,1] -> G
            #   image0[:,:,2] -> B


        ### read labeling
        labelmap0 = imageio.imread(fname_full_labelmap)     #   labelmap0: Array, uint8
        labelmap0 = np.array(labelmap0, dtype=np.uint8)
            # completed to set
            #   labelmap0: ndarray (360, 480), uint8, val 0~18, 255



        #############################################################################################
        # Note that the following things should be considered at this moment:
        #   - resizing of image and labelmap
        #   - 255 in labelmap0 should be converted into 250
        #############################################################################################

        ###------------------------------------------------------------------------------------------
        ### resize
        ###------------------------------------------------------------------------------------------
        image_b = 0
        labelmap_b = 0

        # note that the followings are related to resizing
        #       self.b_resize, self.size_img_rsz[0]: height, self.size_img_rsz[1]: width
        if self.b_resize is True:
            ###
            h_rsz = self.size_img_rsz[0]
            w_rsz = self.size_img_rsz[1]

            ### resize image & labelmap
            image_rsz = np.array( Image.fromarray(image0).resize((w_rsz, h_rsz)), dtype=np.uint8)
            labelmap_rsz = np.array( Image.fromarray(labelmap0).resize((w_rsz, h_rsz)), dtype=np.uint8)
                # image_rsz   : (h,w,c), uint8
                # labelmap_rsz: (h,w),   uint8, val 0~18, 255

            ###
            image_b    = copy.deepcopy(image_rsz)
            labelmap_b = copy.deepcopy(labelmap_rsz)
        else:
            ###
            image_b    = copy.deepcopy(image0)
            labelmap_b = copy.deepcopy(labelmap0)
        #end
            # completed to set
            #       image_b:
            #       labelmap_b


        ###------------------------------------------------------------------------------------------
        ### post-processing of labeling
        ###------------------------------------------------------------------------------------------
        # note that labels other than 0~18 should be 250 (which indicates invalid)

        set_idx_invalid = (labelmap_b > 18)
            # completed to set
            #       set_idx_invalid: (h, w), bool

        labelmap_b[set_idx_invalid] = 250
            # completed to set
            #       labelmap_b: (h, w), uint8


        ###------------------------------------------------------------------------------------------
        ### apply augmentation (if necessary)
        ###------------------------------------------------------------------------------------------
        if self.augmentations is not None:
            image_b, labelmap_b = self.augmentations(image_b, labelmap_b)
        #end
            # completed to set
            #   image_b: ndarray (360, 480, 3), val 0~255
            #   labelmap_b: ndarray (360, 480), val 0~18, 250
            #              (Note that the default value seems 250
            #               corresponding to some unfilled region due to the augmentation.)


        ###------------------------------------------------------------------------------------------
        ### apply transform (if necessary)
        ###------------------------------------------------------------------------------------------
        ### Here, transform means conversion of numpy into torch Tensor.
        if self.b_do_transform:
            image_b, labelmap_b = self.transform(image_b, labelmap_b)
        #end
            # completed to set
            #   image_b: torch.Size([3,360,480]), val 0.0 ~ 1.0   (however, it can shift, e.g. -0.3 ~ 0.7)
            #   labelmap_b: torch.Size([360,480]), val 0~18, 250


        return image_b, labelmap_b
            # completed to
            #   image0: torch.Size([3,360,480])
            #   labelmap0: torch.Size([360,480])

    #==================================================================================================================
    def transform(self, image, labelmap):
        ###===================================================================================================
        ### transform (numpy -> torch Tensor) [only called in __getitem__()]
        ###===================================================================================================
        # <before transform> [input of this function]
        #   img: ndarray (360, 480, 3), val 0~255
        #   lbl: ndarray (360, 480), val 0~18, 250
        #
        # <after transform> [output of this function]
        #   img: torch.Size([3,360,480]), val 0.0 ~ 1.0   (however, it can shift, e.g. -0.3 ~ 0.7)
        #   lbl: torch.Size([360,480]), val 0~18, 250


        ### (1) convert RGB -> BGR
        image = image[:, :, ::-1]  # RGB -> BGR


        ### (2) apply mean-offsetting
        image = image.astype(np.float64)
        image -= self.rgb_mean


        ### (3) do value scaling (resize scales images from 0 to 255, thus we need to divide by 255.0)
        #if self.img_norm:
        if 1:
            image = image.astype(float) / 255.0
        #end


        ### (4) convert HWC -> CHW
        image = image.transpose(2, 0, 1)


        ### (5) convert to torch Tensor
        image    = torch.from_numpy(image).float()
        labelmap = torch.from_numpy(labelmap).long()


        return image, labelmap
    #==================================================================================================================
    def decode_segmap(self, labelmap, plot=False):
        ###===================================================================================================
        ### convert label_map into visible_img [only called in __main__()]
        ###===================================================================================================

        # labelmap: label_map, ndarray (360, 480)


        ###------------------------------------------------------------------------------------------
        ### setting
        ###------------------------------------------------------------------------------------------

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
        r = np.ones_like(labelmap)*250          # 250: indicating invalid label
        g = np.ones_like(labelmap)*250
        b = np.ones_like(labelmap)*250

        for l in range(0, self.n_classes):
            ### find
            idx_set = (labelmap == l)           # idx_set: ndarray, (h, w), bool

            ### assign
            r[idx_set] = rgb_labels[l, 0]       # r: 0 ~ 255
            g[idx_set] = rgb_labels[l, 1]       # g: 0 ~ 255
            b[idx_set] = rgb_labels[l, 2]       # b: 0 ~ 255
        #end

        img_label_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3))
        img_label_rgb[:, :, 0] = r / 255.0
        img_label_rgb[:, :, 1] = g / 255.0
        img_label_rgb[:, :, 2] = b / 255.0

        return img_label_rgb

########################################################################################################################
# [Note] (by Jungwon)
#   RailSem19Loader()
#       __init__()
#       __len__()           : returns the total number of imgs (e.g. total OOOO training imgs)
#       __getitem__()
#           transform()     : only called in __getitem__()
#       decode_segmap()     : only called in  __main__()
#
# *installed imageio for imread
########################################################################################################################
if __name__ == "__main__":
    ###============================================================================================================
    ### setting
    ###============================================================================================================

    ### (1) set path for dataset
    path_root = "/media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417"
        # completed to set
        #       path_root
        #---------------------------------------------------------------------------------------------
        # note that there are the following folders:
        #   /media/yu1/hdd_my/Dataset_railsem19/rs19_val_20200417
        #       : fpath_root
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
    trainloader_head = RailSem19Loader(path_root, b_do_transform=True)
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
                img_labelmap = trainloader_head.decode_segmap(labelmap_np)      # img_labelmap: ndarray (360, 480, 3)

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


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

    #local_path = "/home/meetshah1995/datasets/segnet/CamVid"
    #local_path = "/media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid"
        #---------------------------------------------------------------------------------------------
        # note that there are following folders including (480 x 360) images
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/test       (233 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/testannot  (233 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/train      (367 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/trainannot (367 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/val        (101 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/valannot   (101 imgs)
        # ---------------------------------------------------------------------------------------------


########################################################################################################################

        # if not self.test_mode:
        #     for split in ["train", "test", "val"]:
        #         file_list = os.listdir(root + "/" + split)
        #         self.fnames[split] = file_list
        #     #end
        # #end
            # completed to set
            #   self.fnames[]: list for containing img-names only
            #                 (e.g. 'aaa01.png', 'aaa02.png', ...)

########################################################################################################################

        #self.mean = np.array([104.00699, 116.66877, 122.67892])
        #self.mean = np.array([0.0, 0.0, 0.0])



        ### (1) resize (note that self.img_size[0]: 360, self.img_size[1]: 480)
        #img = np.array(Image.fromarray(img).resize((self.img_size[1], self.img_size[0])))       # resize(width, height)
            # completed to set
            #   img: ndarray, (360, 480, 3)

