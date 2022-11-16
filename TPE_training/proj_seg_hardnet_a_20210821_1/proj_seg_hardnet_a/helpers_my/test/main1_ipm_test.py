# 2019/8/2
# Jungwon Kang


# Using cv2.getPerspectiveTransform

# <references>
#   https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#   https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#   Using newaxis -> https://medium.com/@ian.dzindo01/what-is-numpy-newaxis-and-when-to-use-it-8cb61c7ed6ae


# four corners in the test img: these four points covers a pair of two rails
#   <1>     <4>
#   <2>     <3>
#
#       x       y
# <1>   898     421
# <2>   380     1215
# <3>   1315    1215
# <4>   1142    421

# Here, we transform these four points into a certain rectangle.

import os
import cv2              # 4.1.0
import numpy as np

########################################################################################################################
### function: transform cam to bev
########################################################################################################################
def transform_cam_to_bev(z_cam, x_cam, param_scale_pixel_m, param_offset_y, param_offset_x):
    y_img_bev = -1.0*param_scale_pixel_m*z_cam + param_offset_y
    x_img_bev = param_scale_pixel_m*x_cam + param_offset_x

    return x_img_bev, y_img_bev
#end

########################################################################################################################
### function: get transform
########################################################################################################################
def get_transform(pnt_in_img_in1, pnt_in_img_in2, pnt_in_img_in3, pnt_in_img_in4,
                  pnt_in_cam1, pnt_in_cam2, pnt_in_cam3, pnt_in_cam4, param_scale_pixel_m, param_offset_y, param_offset_x):
    ###
    pnt_in_bev1 = transform_cam_to_bev(pnt_in_cam1[1], pnt_in_cam1[0],
                                       param_scale_pixel_m, param_offset_y, param_offset_x)
    pnt_in_bev2 = transform_cam_to_bev(pnt_in_cam2[1], pnt_in_cam2[0],
                                       param_scale_pixel_m, param_offset_y, param_offset_x)
    pnt_in_bev3 = transform_cam_to_bev(pnt_in_cam3[1], pnt_in_cam3[0],
                                       param_scale_pixel_m, param_offset_y, param_offset_x)
    pnt_in_bev4 = transform_cam_to_bev(pnt_in_cam4[1], pnt_in_cam4[0],
                                       param_scale_pixel_m, param_offset_y, param_offset_x)

    ### compose pnts
    pnts_src = np.float32([pnt_in_img_in1, pnt_in_img_in2, pnt_in_img_in3, pnt_in_img_in4])
    pnts_dst = np.float32([pnt_in_bev1, pnt_in_bev2, pnt_in_bev3, pnt_in_bev4])
    # Note that pnt order does not matter.
    #   For example, [pnt_src3, pnt_src2, pnt_src4, pnt_src1] also works.

    ### get transform
    m_tf = cv2.getPerspectiveTransform(pnts_src, pnts_dst)

    return m_tf
#end

########################################################################################################################
### user setting
########################################################################################################################
pnt_in_img_in1 = [469, 270]     # x,y
pnt_in_img_in2 = [283, 539]
pnt_in_img_in3 = [541, 539]
pnt_in_img_in4 = [485, 270]

pnt_in_cam1 = [-0.75, 25.0]
pnt_in_cam2 = [-0.75, 0.0]
pnt_in_cam3 = [ 0.75, 0.0]
pnt_in_cam4 = [ 0.75, 25.0]

param_scale_pixel_m = 30.0          # oo pixel/meter
param_h_img_bev = 720
param_w_img_bev = 480

param_offset_y = param_h_img_bev
param_offset_x = param_w_img_bev/2.0


###
param_h_img_rsz = 540
param_w_img_rsz = 960
param_dir_input = "/media/yu1/hdd_my/Dataset_YDHR_OCT009_OCT10/img_gopro/selected/test7_run2_normal_reverse_switch"
param_dir_output = "/home/yu1/Desktop/dir_temp/bev_test"

###
b_show_img_raw = 0
b_show_img_dst = 1



########################################################################################################################
### init
########################################################################################################################
m_tf = get_transform(pnt_in_img_in1, pnt_in_img_in2, pnt_in_img_in3, pnt_in_img_in4,
                     param_scale_pixel_m, param_offset_y, param_offset_x)


########################################################################################################################
### loop
########################################################################################################################
print("Process all images inside : {}".format(param_dir_input))

list_fnames_img_ = os.listdir(param_dir_input)
list_fnames_img  = sorted(list_fnames_img_)


for fname_img_in in list_fnames_img:
    ###================================================================================================
    ###
    ###================================================================================================
    print("Process {}".format(fname_img_in))
    full_fname_img = os.path.join(param_dir_input, fname_img_in)


    ###================================================================================================
    ###
    ###================================================================================================
    img_raw_ori = cv2.imread(full_fname_img)
    img_raw_rsz = cv2.resize(img_raw_ori, (param_w_img_rsz, param_h_img_rsz))


    if b_show_img_raw:
        cv2.imshow("img_raw_rsz", img_raw_rsz)
        cv2.waitKey(1)
    #end



    ###================================================================================================
    ### transform (img_raw_rsz to img_dst)
    ###================================================================================================
    img_dst = cv2.warpPerspective(img_raw_rsz, m_tf, (param_w_img_bev, param_h_img_bev))


    ### save & show
    fname_img_dst_out = param_dir_output + '/bev_' + fname_img_in

    cv2.imwrite(fname_img_dst_out, img_dst)
    cv2.waitKey(1)


    if b_show_img_dst:
        cv2.imshow("img_dst", img_dst)
        cv2.waitKey(1)
    #end
#end

print("The END")




########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################



"""
pnt_in_img_in1 = [469*4, 270*4]     # x,y
pnt_in_img_in2 = [283*4, 539*4]
pnt_in_img_in3 = [541*4, 539*4]
pnt_in_img_in4 = [485*4, 270*4]

pnt_in_cam1 = [-0.75, 25.0]
pnt_in_cam2 = [-0.75, 0.0]
pnt_in_cam3 = [ 0.75, 0.0]
pnt_in_cam4 = [ 0.75, 25.0]

param_scale_pixel_m = 30.0          # oo pixel/meter
param_h_img_bev = 720
param_w_img_bev = 480

param_offset_y = param_h_img_bev
param_offset_x = param_w_img_bev/2.0


###
param_h_img_rsz = 2160
param_w_img_rsz = 3840
param_dir_input = "/media/yu1/hdd_my/Dataset_YDHR_OCT009_OCT10/img_gopro/selected/test7_run2_normal_reverse_switch"
param_dir_output = "/home/yu1/Desktop/dir_temp/bev_test"

###
b_show_img_raw = 0
b_show_img_dst = 1
"""
