import os
import collections
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate

### <newly added>
import imageio                      # for using imageio.imread() instead of m.imread()
from PIL import Image

# 2020/03/30
# Modified by Jungwon


class camvidLoader(data.Dataset):
    #==================================================================================================================
    def __init__(self, root, split="train", is_transform=False, img_size=None, augmentations=None, img_norm=True, test_mode=False):
        self.root = root
        self.split = split
        self.is_transform = is_transform

        #if not img_size:
        #    self.img_size = [360, 480]
        #end
        self.img_size = [360, 480]

        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        #self.mean = np.array([0.0, 0.0, 0.0])
        self.n_classes = 12
        self.files = collections.defaultdict(list)

        if not self.test_mode:
            for split in ["train", "test", "val"]:
                file_list = os.listdir(root + "/" + split)
                self.files[split] = file_list
            #end
        #end
            # completed to set
            #   self.files[]: list for containing img-names only
            #                 (e.g. 'aaa01.png', 'aaa02.png', ...)

        # completed to set
        #   self.root           : root path for dataset
        #   self.split          : dataset type or additional path for dataset (e.g. /train, /test, /val)
        #   self.img_size       : img size
        #   self.is_transform   : on/off for doing transform
        #   self.augmentations  : on/off for doing augmentations
        #   self.img_norm       : on/off for doing img normalization
        #   self.test_mode      : acts the same as 'self.split'
        #   self.files[]        : list for containing img-names only


    #==================================================================================================================
    def __len__(self):
        return len(self.files[self.split])
        # note that there are 367 training imgs.
        # 'len(self.files[self.split])' should be 367.
    #==================================================================================================================
    def __getitem__(self, index):
        ###===================================================================================================
        ### read ONE img & annotation
        ###===================================================================================================

        # img: image
        # lbl: label (annotation)

        ###------------------------------------------------------------------------------------------
        ### read rgb img & corresponding annotation
        ###------------------------------------------------------------------------------------------

        ### set fname & path
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        lbl_path = self.root + "/" + self.split + "annot/" + img_name
            # completed to set
            #   img_path
            #   lbl_path


        ### read rgb img
        img = imageio.imread(img_path)
        img = np.array(img, dtype=np.uint8)
            # completed to set
            #   img: ndarray (360, 480, 3), val 0~255


        ### read annotation
        lbl = imageio.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int8)
            # completed to set
            # lbl: ndarray (360, 480), val 0~11


        ###------------------------------------------------------------------------------------------
        ### apply augmentation and/or transform
        ###------------------------------------------------------------------------------------------

        ### apply augmentation
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        #end
            # completed to set
            #   img: ndarray (360, 480, 3), val 0~255
            #   lbl: ndarray (360, 480), val 0~250 (but, actually val 0~11)
            #        (Note that the default value seems 250
            #         corresponding to some unfilled region due to the augmentation.)


        ### apply transform (Here, transform means conversion of numpy into torch Tensor.)
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        #end
            # completed to set
            #   img: torch.Size([3,360,480]), val 0.0 ~ 1.0   (however, it can shift, e.g. it can have e.g. -0.3 ~ 0.7)
            #   lbl: torch.Size([360,480]), val 0 ~ 250       (but, actually val 0~11)


        return img, lbl
            # completed to
            #   img: torch.Size([3,360,480])
            #   lbl: torch.Size([360,480])

    #==================================================================================================================
    def transform(self, img, lbl):
        ###===================================================================================================
        ### transform (numpy -> torch Tensor) [only called in __getitem__()]
        ###===================================================================================================
        # <before transform> [input of this function]
        #   img: ndarray (360, 480, 3), val 0~255
        #   lbl: ndarray (360, 480), val 0~250 (but, actually val 0~11)
        #
        # <after transform> [output of this function]
        #   img: torch.Size([3,360,480]), val 0.0 ~ 1.0   (however, it can shift, e.g. it can have e.g. -0.3 ~ 0.7)
        #   lbl: torch.Size([360,480]), val 0 ~ 250 (but, actually val 0~11)


        ### (1) resize (note that self.img_size[0]: 360, self.img_size[1]: 480)
        img = np.array(Image.fromarray(img).resize((self.img_size[1], self.img_size[0])))       # resize(width, height)
            # completed to set
            #   img: ndarray, (360, 480, 3)


        ### (2) convert RGB -> BGR
        img = img[:, :, ::-1]  # RGB -> BGR


        ### (3) apply mean-offsetting
        img = img.astype(np.float64)
        img -= self.mean


        ### (4) do value scaling (resize scales images from 0 to 255, thus we need to divide by 255.0)
        if self.img_norm:
            img = img.astype(float) / 255.0
        #end


        ### (5) convert HWC -> CHW
        img = img.transpose(2, 0, 1)


        ### (6) convert to torch Tensor
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
    #==================================================================================================================
    def decode_segmap(self, temp, plot=False):
        ###===================================================================================================
        ### convert label_map into visible_img [only called in __main__()]
        ###===================================================================================================

        # temp: label_map, ndarray (360, 480)


        ###------------------------------------------------------------------------------------------
        ### setting
        ###------------------------------------------------------------------------------------------

        ###
        Sky = [128, 128, 128]           # class0
        Building = [128, 0, 0]          # class1
        Pole = [192, 192, 128]          # class2
        Road = [128, 64, 128]           # class3
        Pavement = [60, 40, 222]        # class4
        Tree = [128, 128, 0]            # class5
        SignSymbol = [192, 128, 128]    # class6
        Fence = [64, 64, 128]           # class7
        Car = [64, 0, 128]              # class8
        Pedestrian = [64, 64, 0]        # class9
        Bicyclist = [0, 128, 192]       # class10
        Unlabelled = [0, 0, 0]          # class11

        ###
        label_colours = np.array(
            [
                Sky,                    # class0
                Building,               # class1
                Pole,                   # class2
                Road,                   # class3
                Pavement,               # class4
                Tree,                   # class5
                SignSymbol,             # class6
                Fence,                  # class7
                Car,                    # class8
                Pedestrian,             # class9
                Bicyclist,              # class10
                Unlabelled,             # class11
            ]
        )


        ###------------------------------------------------------------------------------------------
        ### create visible_img
        ###------------------------------------------------------------------------------------------
        r = np.ones_like(temp)*255
        g = np.ones_like(temp)*255
        b = np.ones_like(temp)*255

        for l in range(0, self.n_classes):
            ### find
            idx_set = (temp == l)                   # idx_set: ndarray, bool (360, 480)

            ### assign
            r[idx_set] = label_colours[l, 0]      # r: 0 ~ 255
            g[idx_set] = label_colours[l, 1]      # g: 0 ~ 255
            b[idx_set] = label_colours[l, 2]      # b: 0 ~ 255
        #end

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb

########################################################################################################################
# [Note] (by Jungwon)
#   camvidLoader()
#       __init__()
#       __len__()       : returns the total number of imgs (e.g. 367 for training imgs)
#       __getitem__()
#       transform()     : only called in __getitem__()
#       decode_segmap() : only called in  __main__()
#
# *installed imageio for imread
########################################################################################################################
if __name__ == "__main__":
    ###============================================================================================================
    ### setting
    ###============================================================================================================

    ### (1) set path for dataset
    #local_path = "/home/meetshah1995/datasets/segnet/CamVid"
    local_path = "/media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid"
        # completed to set
        #       local path
        #---------------------------------------------------------------------------------------------
        # note that there are following folders including (480 x 360) images
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/test       (233 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/testannot  (233 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/train      (367 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/trainannot (367 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/val        (101 imgs)
        #   ...../camvid_b/SegNet-Tutorial-master/CamVid/valannot   (101 imgs)
        # ---------------------------------------------------------------------------------------------


    ### (2) create an object for augmentations
    augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip(0.5)])
        # completed to create
        #       augmentations


    ### (3) set batch-size for data loading
    bs = 4      # bs: batch_size
        # completed to set
        #       bs


    ###============================================================================================================
    ### create objects for dataloader
    ###============================================================================================================

    ### (1) create an object1 for dataloader
    dst = camvidLoader(local_path, is_transform=True, augmentations=augmentations)
        # completed to create
        #       dst


    ### (2) create an object2 for dataloader
    trainloader = data.DataLoader(dst, batch_size=bs)
        # completed to create
        #       trainloader


    ###============================================================================================================
    ### loop
    ###============================================================================================================

    ### (1) create fig
    fig_plt, axarr = plt.subplots(bs, 2)
        # completed to set
        #       fig_plt: fig object
        #       axarr:   axes object

    ### (2) loop
    for i, data_samples in enumerate(trainloader):
        # i: 0 ~ 91
        #   note that there are 367 training images, which means that there are idx: 0 ~ 366 for training imgs
        #       i -> (batch-size*i) ~ (batch-size*(i+1) - 1) for idx of training imgs
        #       i:0 -> 0 ~ 3
        #       i:91 -> 364 ~ 367
        # data_samples: list from trainloader
        #       [0]: imgs with size batch-size
        #       [1]: labels with size batch-size

        print('showing {}'.format(i))

        ### load imgs & labels
        imgs, labels = data_samples

        imgs = imgs.numpy()[:, ::-1, :, :]
            # imgs: ndarray, (bs, 3, 360, 480)

        imgs = np.transpose(imgs, [0, 2, 3, 1])
            # imgs: ndarray, (bs, 360, 480, 3)


        if 1:
            ### show
            for j in range(bs):
                axarr[j][0].imshow(imgs[j])

                nd_lbl = labels.numpy()[j]              # nd_lbl: ndarray (360, 480), val 0~250 (but, actually val 0~11)
                img_lbl = dst.decode_segmap(nd_lbl)     # img_lbl: ndarray (360, 480, 3)
                axarr[j][1].imshow(img_lbl)

                #axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
            #end

            #plt.show()
            str_tit = '%d' % i
            fig_plt.suptitle(str_tit)
            plt.draw()
            plt.pause(0.01)
        #end
    #end


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

        """
        a = input()
        print('next..................')
        if a == "ex":
            break
        else:
            plt.close()
        #end
        """


        #plt.close()


        #img = m.imread(img_path)               # -> deprecated
        #lbl = m.imread(lbl_path)               # -> deprecated

        #img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode     # -> deprecated
