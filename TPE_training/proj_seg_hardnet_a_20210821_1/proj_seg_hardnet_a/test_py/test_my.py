import os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import copy

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True
import cv2


## revised by Jungwon (May 1 2020)

########################################################################################################################
def init_dataloader(args):

    ###
    data_loader = get_loader(args.dataset)
    loader = data_loader(root=None, is_transform=True, img_size=eval(args.size), test_mode=True)
    n_classes = loader.n_classes

    return loader, n_classes
########################################################################################################################
def init_model(args, n_classes):

    ### setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = get_model({"arch": "hardnet"}, n_classes)
    # state = convert_state_dict(torch.load(args.model_path, map_location=device)["model_state"])
    # model.load_state_dict(state)
    # model.eval()
    # model.to(device)


    model_dict = {"arch": "hardnet"}
    model = get_model(model_dict, n_classes, version=args.dataset)
    #aaa = torch.load(args.model_path)["model_state"]
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)



    return device, model


########################################################################################################################
def process_img_my(img_path, size, device, model, loader):
    print("Read Input Image from : {}".format(img_path))

    # img = cv2.imread(img_path)
    # img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    # img = img_resized.astype(np.float16)
    #
    # # norm
    # value_scale = 255
    # mean = [0.406, 0.456, 0.485]
    # mean = [item * value_scale for item in mean]
    # std = [0.225, 0.224, 0.229]
    # std = [item * value_scale for item in std]
    # img = (img - mean) / std

    img = misc.imread(img_path)
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    img_resized = copy.deepcopy(img)

    img = img[:, :, ::-1]       # rgb -> bgr
    img = img.astype(np.float64)
    img -= loader.mean

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
    decoded = loader.decode_segmap(pred)

    return img_resized, decoded


########################################################################################################################
def process_img(img_path, size, device, model, loader):
    print("Read Input Image from : {}".format(img_path))

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    img = img_resized.astype(np.float16)

    # norm
    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)

    return img_resized, decoded

########################################################################################################################
def test(args):

    ###============================================================================================================
    ###
    ###============================================================================================================
    loader, n_classes = init_dataloader(args)
    device, model = init_model(args, n_classes)
    proc_size = eval(args.size)


    ###============================================================================================================
    ###
    ###============================================================================================================
    if os.path.isfile(args.input):
        ### IF input is a file:

        ###
        img_raw, decoded = process_img_my(args.input, proc_size, device, model, loader)
            # completed to set
            #       img_raw
            #       decoded

        blend = np.concatenate((img_raw, decoded), axis=1)
        out_path = os.path.join(args.output, os.path.basename(args.input))
        #cv2.imwrite("test.png", decoded)
        #cv2.imwrite(out_path, blend)
        misc.imsave(out_path, decoded)


    elif os.path.isdir(args.input):
        ### IF input is a dir:
        print("Process all image inside : {}".format(args.input))

        for img_file in os.listdir(args.input):
            ###
            _, ext = os.path.splitext(os.path.basename((img_file)))
            if ext not in [".png", ".jpg"]:
                continue
            #end

            img_path = os.path.join(args.input, img_file)

            ###
            img, decoded = process_img_my(img_path, proc_size, device, model, loader)
            blend = np.concatenate((img, decoded), axis=1)
            out_path = os.path.join(args.output, os.path.basename(img_file))
            cv2.imwrite(out_path, blend)
        #end
    #end


########################################################################################################################
# parameters:
#   --model_path /home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/hardnet_camvid_my/cur_my/hardnet_camvid_best_model.pkl
#   --dataset camvid
#   --size 360,480
#   --input /media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid/test
#   --input /media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid/test/0001TP_008550.png
#   --output /home/yu1/Desktop/temp2
#
########################################################################################################################
#   --model_path /home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/hardnet_camvid_my/cur_my/hardnet_camvid_best_model.pkl --dataset camvid --size 360,480 --input /media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid/test --input /media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid/test/0001TP_008550.png --output /home/yu1/Desktop/temp2
########################################################################################################################
if __name__ == "__main__":
    ###============================================================================================================
    ### (1) set parser
    ###============================================================================================================
    parser = argparse.ArgumentParser(description="Params")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="icboard",
        help="Path to the dataset",
    )   # -> dataset is needed to convert a estimated label map into RGB result image.

    parser.add_argument(
        "--size",
        type=str,
        default="540,960",
        help="Inference size",
    )

    parser.add_argument(
        "--input",
        nargs="?",
        type=str,
        default=None,
        help="Path of the input image/ directory"
    )

    parser.add_argument(
        "--output",
        nargs="?",
        type=str,
        default="./",
        help="Path of the output directory"
    )

    args = parser.parse_args()


    ###============================================================================================================
    ### (2) test
    ###============================================================================================================
    test(args)


########################################################################################################################

#--model_path /home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/hardnet_camvid_my/cur_my/hardnet_camvid_best_model.pkl --dataset camvid --size 360,480 --input /media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid/test --input /media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid/test/0001TP_008550.png --output /home/yu1/Desktop/temp2

#--model_path /home/yu1/PycharmProjects/proj_seg_hardnet_a/runs/hardnet_camvid_my/cur/hardnet_camvid_best_model.pkl --dataset camvid --size 360,480 --input /media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid/test --input /media/yu1/hdd_my/Dataset_camvid/camvid_b/SegNet-Tutorial-master/CamVid/test/0001TP_008550.png --output /home/yu1/Desktop/temp2

