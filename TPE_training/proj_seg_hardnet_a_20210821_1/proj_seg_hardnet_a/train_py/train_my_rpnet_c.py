# 2020/7/12
# Jungwon Kang

import os
import math
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from scipy.signal import find_peaks

import sys
import cv2

from torch.utils                import data
from tqdm                       import tqdm

from ptsemseg.models            import get_model
from ptsemseg.loss              import get_loss_function   ## Segmentation Loss
from ptsemseg.loader            import get_loader
from ptsemseg.utils             import get_logger
from ptsemseg.metrics           import runningScore, averageMeter
from ptsemseg.augmentations     import get_composed_augmentations
from ptsemseg.schedulers        import get_scheduler
from ptsemseg.optimizers        import get_optimizer

from helpers_my                 import my_utils
from helpers_my                 import my_loss             ## Center/LeftRight Loss
from ptsemseg.loader.myhelpers  import myhelper_railsem19

from tensorboardX               import SummaryWriter



########################################################################################################################
### train()
########################################################################################################################
def train(cfg, writer, logger, fname_weight_init):
    ###============================================================================================================
    ### (0) init
    ###============================================================================================================

    ### setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    ### setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device: cuda (if gpu is available)


    ###============================================================================================================
    ### (1) create dataloader
    ###============================================================================================================
    data_loader   = get_loader(cfg["data"]["dataset"])


    ###---------------------------------------------------------------------------------------------------
    ### create dataloader_head
    ###---------------------------------------------------------------------------------------------------
    t_loader_head = data_loader(type_trainval="train", output_size_hmap="size_img_rsz")
    v_loader_head = data_loader(type_trainval="val",   output_size_hmap="size_img_rsz")
        # completed to create
        #       t_loader_head
        #       v_loader_head


    ###---------------------------------------------------------------------------------------------------
    ### create t_loader_batch
    ###---------------------------------------------------------------------------------------------------
    t_loader_batch = data.DataLoader(
        t_loader_head,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=False,
        )
        # completed to create
        #       t_loader_batch


    ###---------------------------------------------------------------------------------------------------
    ### create v_loader_batch
    ###---------------------------------------------------------------------------------------------------
    v_loader_batch = data.DataLoader(
        v_loader_head,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"]
        )
        # completed to create
        #       v_loader_batch


    ###---------------------------------------------------------------------------------------------------
    ### set n_classes_seg
    ###---------------------------------------------------------------------------------------------------
    n_classes_seg = t_loader_head.n_classes_seg
    n_classes_ins = t_loader_head.n_classes_ins

    ###============================================================================================================
    ### (2) setup model
    ###============================================================================================================

    ###---------------------------------------------------------------------------------------------------
    ### setup model
    ###---------------------------------------------------------------------------------------------------
    model = get_model(cfg["model"], n_classes_ins, n_classes_seg).to(device)

    # total_params = sum(p.numel() for p in model.parameters())
    # print( 'Parameters:',total_params )


    ###---------------------------------------------------------------------------------------------------
    ### init weights
    ###---------------------------------------------------------------------------------------------------
    model.apply(my_utils.weights_init)


    ###---------------------------------------------------------------------------------------------------
    ### load initial (pre-trained) weights into model
    ###---------------------------------------------------------------------------------------------------
    my_utils.load_weights_to_model(model, fname_weight_init)


    ###---------------------------------------------------------------------------------------------------
    ### freeze model of only FC-HarDNet part
    ###---------------------------------------------------------------------------------------------------
    if 0:
        ###
        for param_this in model.base.parameters():
            param_this.requires_grad = False
        #end

        ###
        for param_this in model.conv1x1_up.parameters():
            param_this.requires_grad = False
        #end

        ###
        for param_this in model.denseBlocksUp.parameters():
            param_this.requires_grad = False
        #end

        ###
        model.finalConv.bias.requires_grad = False
        model.finalConv.weight.requires_grad = False

        ###
        for param_this in model.transUpBlocks.parameters():
            param_this.requires_grad = False
        #end
    #end



    ###============================================================================================================
    ### (3) setup optimizer
    ###============================================================================================================
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    #print("Using optimizer {}".format(optimizer))
    logger.info("Using optimizer {}".format(optimizer))



    ###============================================================================================================
    ### (4) setup lr_scheduler
    ###============================================================================================================
    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])



    ###============================================================================================================
    ### (5) setup loss
    ###============================================================================================================
    loss_fn = get_loss_function(cfg)

    #print("Using loss {}".format(loss_fn))
    logger.info("Using loss {}".format(loss_fn))



    ###============================================================================================================
    ### (6) init for training
    ###============================================================================================================

    ### setup metrics
    running_metrics_val = runningScore(n_classes_seg)

    ### init
    start_iter = 0

    ###
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    ###
    best_iou = -100.0
    best_loss_hmap = 1000000000.0
    i = start_iter
    flag = True

    ###
    loss_accum_all = 0
    loss_accum_var = 0
    loss_accum_dis = 0
    loss_accum_reg = 0
    loss_accum_seg = 0
    loss_accum_centerline = 0

    num_loss = 0

    ###============================================================================================================
    ### (7) loop for training
    ###============================================================================================================
    while i <= cfg["training"]["train_iters"] and flag:
        for data_batch in t_loader_batch:
            ###
            i += 1
            start_ts = time.time()


            ###
            imgs_raw_fl_n                        = data_batch['img_raw_fl_n']                     # (bs, 3, h_rsz, w_rsz)
            gt_imgs_label_seg                    = data_batch['gt_img_label_seg']                 # (bs, h_rsz, w_rsz)
            gt_ins_pose                          = data_batch['gt_instances']
            gt_labelmap_centerline               = data_batch['gt_labelmap_centerline']           # (bs, 1, h_rsz, w_rsz)
            gt_labelmap_leftright                = data_batch['gt_labelmap_leftright']            # (bs, 2, h_rsz, w_rsz)


            imgs_raw_fl_n           = imgs_raw_fl_n.to(device)
            gt_imgs_label_seg       = gt_imgs_label_seg.to(device)
            gt_ins_pose             = gt_ins_pose.to(device)
            gt_labelmap_centerline  = gt_labelmap_centerline.to(device)
            gt_labelmap_leftright   = gt_labelmap_leftright.to(device)

            ###
            scheduler.step()
            model.train()
            optimizer.zero_grad()

            ###
            output_instance, outputs_seg, outputs_centerline = model(imgs_raw_fl_n)
            ###============================================================================================================
            ### (8) Loss
            ###============================================================================================================
            loss_instance, dis_loss, reg_loss, var_loss = my_loss.Discriminative_loss(output_instance,gt_ins_pose,0.5,3)
            loss_seg = loss_fn(input=outputs_seg, target=gt_imgs_label_seg, train_val=0)
            loss_centerline = my_loss.MSE_loss(x_est=outputs_centerline, x_gt=gt_labelmap_centerline, b_sigmoid=True)
            loss_this = loss_instance + loss_seg + 0.08 * loss_centerline

            ###
            loss_this.backward()
            optimizer.step()

            ###
            c_lr = scheduler.get_lr()

            ###
            time_meter.update(time.time() - start_ts)

            ###
            loss_accum_all += loss_this.item()
            loss_accum_var += var_loss.item()
            loss_accum_dis += dis_loss.item()
            loss_accum_reg += reg_loss.item()
            loss_accum_seg        += loss_seg.item()
            loss_accum_centerline += loss_centerline.item()


            num_loss += 1


            ### print (on demand)
            if (i + 1) % cfg["training"]["print_interval"] == 0:
                ###
                fmt_str = "Iter [{:d}/{:d}]  Loss (all): {:.7f}, Loss (var): {:.7f}, Loss (dis): {:.7f}, Loss (reg): {:.7f}, Loss (seg): {:.7f}, Loss (centerline): {:.7f}, lr={:.7f}"

                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_accum_all        / num_loss,
                    loss_accum_var        / num_loss,
                    loss_accum_dis        / num_loss,
                    loss_accum_reg        / num_loss,
                    loss_accum_seg        / num_loss,
                    loss_accum_centerline / num_loss,
                    c_lr[0],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss_this.item(), i + 1)
                time_meter.reset()
            #end


            ################################################################################################
            ### validate (on demand)
            ################################################################################################
            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
                loss_accum_distance_validation    = 0
                loss_accum_variance_validation    = 0
                loss_accum_instance_validation    = 0
                loss_accum_regularizer_validation = 0
                loss_accum_segmentation_validation    = 0
                loss_accum_centerline_validation      = 0


                num_loss_validation = 0
                if (i + 1) % 10000 == 0:
                    for data_batch_validation in v_loader_batch:
                        imgs_raw_fl_n          = data_batch_validation['img_raw_fl_n']                            # (bs, 3, h_rsz, w_rsz)
                        gt_ins_positions       = data_batch_validation['gt_instances']
                        gt_imgs_label_seg      = data_batch_validation['gt_img_label_seg']                        # (bs, h_rsz, w_rsz)
                        gt_labelmap_centerline = data_batch_validation['gt_labelmap_centerline']                  # (bs, 1, h_rsz, w_rsz)
                        gt_labelmap_leftright  = data_batch_validation['gt_labelmap_leftright']

                        imgs_raw_fl_n          = imgs_raw_fl_n.to(device)
                        gt_ins_positions       = gt_ins_positions.to(device)
                        gt_imgs_label_seg      = gt_imgs_label_seg.to(device)
                        gt_labelmap_centerline = gt_labelmap_centerline.to(device)
                        gt_labelmap_leftright  = gt_labelmap_leftright.to(device)


                        v_output_instance, v_output_segmentation, v_output_centerline  =  model(imgs_raw_fl_n)

                        loss_ins, distance_loss, regularizer_loss, variance_loss = my_loss.Discriminative_loss(v_output_instance,gt_ins_positions,0.5,3)
                        seg_loss = loss_fn(input=outputs_seg, target=gt_imgs_label_seg, train_val=1)
                        centerline_loss = my_loss.L1_loss(x_est=outputs_centerline, x_gt=gt_labelmap_centerline, b_sigmoid=True)


                        loss_accum_instance_validation     += loss_ins.item()
                        loss_accum_variance_validation     += variance_loss.item()
                        loss_accum_distance_validation     += distance_loss.item()
                        loss_accum_regularizer_validation  += regularizer_loss.item()
                        loss_accum_segmentation_validation     += seg_loss.item()
                        loss_accum_centerline_validation     += centerline_loss.item()



                        num_loss_validation += 1

                    fmt_str = "(VALIDATION) Iter [{:d}/{:d}], Loss (instance): {:.7f}, Loss (var): {:.7f}, Loss (dis): {:.7f}, Loss (reg): {:.7f}, Loss (segmentation): {:.7f}, Loss (centerline): {:.7f}"

                    print_str = fmt_str.format(
                        i + 1,
                        cfg["training"]["train_iters"],
                        loss_accum_instance_validation / num_loss_validation,
                        loss_accum_variance_validation / num_loss_validation,
                        loss_accum_distance_validation / num_loss_validation,
                        loss_accum_regularizer_validation  / num_loss_validation,
                        loss_accum_segmentation_validation / num_loss_validation,
                        loss_accum_centerline_validation / num_loss_validation
                    )

                    print(print_str)

                #//////////////////////////////////////////////////////////////////////////////////////////
                # temp routine
                #//////////////////////////////////////////////////////////////////////////////////////////

                ###
                state = {"epoch": i + 1,
                         "model_state": model.state_dict(),
                         "best_loss": best_loss_hmap}

                ###
                # save_path = os.path.join(
                #     writer.file_writer.get_logdir(),
                #     "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                # )
                save_path = 'Mybest_' + str(i+1) + '.pkl'

                ###
                torch.save(state, save_path)
                #//////////////////////////////////////////////////////////////////////////////////////////


            #end
            ################################################################################################


            ### check if it reaches max iterations
            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break
            #end
        #end
    #end
#end

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
if __name__ == "__main__":
    ###============================================================================================================
    ### setting
    ###============================================================================================================

    fname_config = '../configs/rpnet_c_railsem19_seg.yml'
    fname_weight_init = '../runs/hardnet/cur/MybestSoFar.pkl'

    ###============================================================================================================
    ### (1) set parser
    ###============================================================================================================
    parser = argparse.ArgumentParser(description="config")
    args   = parser.parse_args()
        # completed to set
        #       args.config: path to config file

    args.config = fname_config


    ###============================================================================================================
    ### (2) open config file
    ###============================================================================================================
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    #end
        # completed to set
        #       cfg: dict containing all the contents of config file


    ###============================================================================================================
    ### (3) create a folder with name of dir_log
    ###============================================================================================================

    ### create fpath for dir_log
    run_id = random.randint(1, 100000)
    # dir_log = os.path.join("/runs", os.path.basename(args.config)[:-4], "cur_20200725")
    dir_log = '/runs/rpnet_c_railsem19_seg/cur_20200725'

        # completed to set
        #       dir_log: 'runs/CONFIG_FILE_NAME/OOOOO'


    ### create a folder with name of dir_log
    writer = SummaryWriter(log_dir=dir_log)
    print("RUNDIR: {}".format(dir_log))


    ### copy config file to dir_log
    shutil.copy(args.config, dir_log)


    ###============================================================================================================
    ### (4) get logger
    ###============================================================================================================
    logger = get_logger(dir_log)
    logger.info("Let's begin...")


    ###============================================================================================================
    ### (5) train
    ###============================================================================================================
    train(cfg, writer, logger, fname_weight_init)


#end
########################################################################################################################




########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


"""
###
torch.cuda.empty_cache()
model.eval()

loss_all = 0
loss_n = 0

###
with torch.no_grad():
    for i_val, data_batch in tqdm(enumerate(v_loader_batch)):
        ###
        imgs_raw_fl_n_val       = data_batch['img_raw_fl_n']
        gt_imgs_label_seg_val   = data_batch['gt_img_label_seg']
        gt_hmap_centerline      = data_batch['gt_hmap_centerline']

        imgs_raw_fl_n_val       = imgs_raw_fl_n_val.to(device)
        gt_imgs_label_seg_val   = gt_imgs_label_seg_val.to(device)
        gt_hmap_centerline = gt_hmap_centerline.to(device)


        ###
        outputs_seg_val, outputs_hmap_centerline_val = model(imgs_raw_fl_n_val)


        ###
        #val_loss = loss_fn(input=outputs_seg, target=gt_imgs_label_seg_val)
        loss_hmap = my_loss.neg_loss(preds=outputs_hmap_centerline, targets=gt_hmap_centerline)

        val_loss = loss_hmap

        #pred = outputs_seg.data.max(1)[1].cpu().numpy()
        #gt = gt_imgs_label_seg_val.data.cpu().numpy()

        #running_metrics_val.update(gt, pred)
        val_loss_meter.update(val_loss.item())
    #end
#end

writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
logger.info("Iter %d Val Loss: %.4f" % (i + 1, val_loss_meter.avg))


# ###
# score, class_iou = running_metrics_val.get_scores()
#
# for k, v in score.items():
#     print(k, v)
#     logger.info("{}: {}".format(k, v))
#     writer.add_scalar("val_metrics/{}".format(k), v, i + 1)
# #end
#
# for k, v in class_iou.items():
#     logger.info("{}: {}".format(k, v))
#     writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)
# #end
#
#
# ###
# val_loss_meter.reset()
# running_metrics_val.reset()


###
# state = {
#         "epoch": i + 1,
#         "model_state": model.state_dict(),
#         "optimizer_state": optimizer.state_dict(),
#         "scheduler_state": scheduler.state_dict(),
#         }
#
# save_path = os.path.join(
#         writer.file_writer.get_logdir(),
#         "{}_{}_checkpoint.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
#         )
#
# torch.save(state, save_path)


# ### check if it is the best one so far
# if score["Mean IoU : \t"] >= best_iou:
#     best_iou = score["Mean IoU : \t"]
#     state = {
#         "epoch": i + 1,
#         "model_state": model.state_dict(),
#         "best_iou": best_iou,
#     }
#     save_path = os.path.join(
#         writer.file_writer.get_logdir(),
#         "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
#     )
#     torch.save(state, save_path)
# #end


### check if it is the best one so far
if (val_loss_meter.avg) <= best_loss_hmap:
    ###
    best_loss_hmap = val_loss_meter.avg

    ###
    state = {"epoch": i + 1,
             "model_state": model.state_dict(),
             "best_loss": best_loss_hmap}

    ###
    save_path = os.path.join(
        writer.file_writer.get_logdir(),
        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
    )

    ###
    torch.save(state, save_path)
#end

torch.cuda.empty_cache()
"""



