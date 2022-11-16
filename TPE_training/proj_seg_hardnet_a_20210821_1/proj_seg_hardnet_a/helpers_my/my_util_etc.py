# 2020/7/12
# Jungwon Kang

# See <helpers_my/test/main1_ipm_test.py>

import os
import cv2              # 4.1.0
import numpy as np



########################################################################################################################
### function: transform cam to bev
########################################################################################################################
def _transform_cam_to_bev(z_cam, x_cam, param_scale_pixel_m, param_offset_y, param_offset_x):
    y_img_bev = -1.0*param_scale_pixel_m*z_cam + param_offset_y
    x_img_bev = param_scale_pixel_m*x_cam + param_offset_x

    return x_img_bev, y_img_bev
#end


########################################################################################################################
### function: get transform
########################################################################################################################
def _get_transform(pnt_in_img_in1, pnt_in_img_in2, pnt_in_img_in3, pnt_in_img_in4,
                   pnt_in_cam1, pnt_in_cam2, pnt_in_cam3, pnt_in_cam4,
                   param_scale_pixel_m, param_offset_y, param_offset_x):

    ###
    pnt_in_bev1 = _transform_cam_to_bev(pnt_in_cam1[1], pnt_in_cam1[0],
                                        param_scale_pixel_m, param_offset_y, param_offset_x)
    pnt_in_bev2 = _transform_cam_to_bev(pnt_in_cam2[1], pnt_in_cam2[0],
                                        param_scale_pixel_m, param_offset_y, param_offset_x)
    pnt_in_bev3 = _transform_cam_to_bev(pnt_in_cam3[1], pnt_in_cam3[0],
                                        param_scale_pixel_m, param_offset_y, param_offset_x)
    pnt_in_bev4 = _transform_cam_to_bev(pnt_in_cam4[1], pnt_in_cam4[0],
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
def get_data_for_ipm():
    ###
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
    m_tf = _get_transform(pnt_in_img_in1, pnt_in_img_in2, pnt_in_img_in3, pnt_in_img_in4,
                          pnt_in_cam1, pnt_in_cam2, pnt_in_cam3, pnt_in_cam4,
                          param_scale_pixel_m, param_offset_y, param_offset_x)


    return m_tf, param_h_img_bev, param_w_img_bev
#end


########################################################################################################################
### loop
########################################################################################################################
if __name__ == "__main__":

    ###=========================================================================================================
    ###
    ###=========================================================================================================
    param_h_img_rsz = 540
    param_w_img_rsz = 960

    param_dir_input = "/media/yu1/hdd_my/Dataset_YDHR_OCT009_OCT10/img_gopro/selected/test7_run2_normal_reverse_switch"
    param_dir_output = "/home/yu1/Desktop/dir_temp/bev_test"



    ###=========================================================================================================
    ###
    ###=========================================================================================================
    m_tf, param_h_img_bev, param_w_img_bev = get_data_for_ipm()



    ###=========================================================================================================
    ###
    ###=========================================================================================================
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


        if 1:
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


        if 1:
            cv2.imshow("img_dst", img_dst)
            cv2.waitKey(1)
        #end
    #end

    print("The END")

#end

