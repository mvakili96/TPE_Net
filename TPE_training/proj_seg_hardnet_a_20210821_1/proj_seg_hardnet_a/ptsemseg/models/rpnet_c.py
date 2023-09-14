# 2020/7/21
# Jungwon Kang

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections


########################################################################################################################
### 1.1 ConvLayer (originated from FC-HarDNet)
########################################################################################################################
# CBR module, where CBR: Conv + BN + ReLU
# h,w,ch_in -> h,w,ch_out
class ConvLayer(nn.Sequential):
    ###======================================================================================================
    ### ConvLayer::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, dilation_this = 1):
        super().__init__()
        if dilation_this == 1:
            self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                              stride=stride, padding=kernel//2, bias = False, dilation = dilation_this))
            self.add_module('norm', nn.BatchNorm2d(out_channels))
            self.add_module('relu', nn.ReLU(inplace=True))
        else:
            self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                                padding='same', bias = False, dilation = dilation_this))
            self.add_module('norm', nn.BatchNorm2d(out_channels))
            self.add_module('relu', nn.ReLU(inplace=True))


        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
    #end


    ###======================================================================================================
    ### ConvLayer::forward()
    ###======================================================================================================
    def forward(self, x):
        return super().forward(x)
    #end
#end



########################################################################################################################
### 1.2 BRLayer (originated from FC-HarDNet)
########################################################################################################################
# # BR module, where BR: BN + ReLU
# # h,w,ch -> h,w,ch
# class BRLayer(nn.Sequential):
#     ###======================================================================================================
#     ### BRLayer::__init__()
#     ###======================================================================================================
#     def __init__(self, in_channels):
#         super().__init__()
#
#         self.add_module('norm', nn.BatchNorm2d(in_channels))
#         self.add_module('relu', nn.ReLU(True))
#     #end
#
#     ###======================================================================================================
#     ### BRLayer::forward()
#     ###======================================================================================================
#     def forward(self, x):
#         return super().forward(x)
#     #end
# #end


########################################################################################################################
### 2. HarDBlock (originated from FC-HarDNet)
########################################################################################################################
class HarDBlock(nn.Module):
    ###======================================================================================================
    ### HarDBlock::get_link()
    ###======================================================================================================
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        #end

        out_channels = growth_rate

        link = []

        for i in range(10):
            dv = 2 ** i

            if layer % dv == 0:
                k = layer - dv
                link.append(k)

                if i > 0:
                    out_channels *= grmul
                #end
            #end
        #end

        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0

        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        #end

        return out_channels, in_channels, link
    #end


    ###======================================================================================================
    ### HarDBlock::get_out_ch()
    ###======================================================================================================
    def get_out_ch(self):
        return self.out_channels
    #end


    ###======================================================================================================
    ### HarDBlock::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, have_dilation = False):
        super().__init__()

        ###
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0       # if upsample else in_channels


        ###
        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out

            if have_dilation:
                layers_.append( ConvLayer(inch, outch, dilation_this=1) )
            else:
                layers_.append(ConvLayer(inch, outch))


            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
            #end


            # print("idx_layers: [%d]" % i)
            # print(link)
            # print("inch: [%d], outch: [%d]" % (inch, outch))
        #end


        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)
    #end


    ###======================================================================================================
    ### HarDBlock::forward()
    ###======================================================================================================
    def forward(self, x):

        layers_ = [x]

        for layer in range(len(self.layers)):
            ###
            link = self.links[layer]
            tin = []


            ###
            for i in link:
                tin.append(layers_[i])
            #end


            ###
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            #end


            ###
            out = self.layers[layer](x)
            layers_.append(out)
        #end

        t = len(layers_)

        out_ = []

        for i in range(t):
            if (i == 0 and self.keepBase) or \
               (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])
            #end
        #end

        out = torch.cat(out_, 1)
        return out
    #end


########################################################################################################################
### 3. TransitionUp (originated from FC-HarDNet)
########################################################################################################################
class TransitionUp(nn.Module):
    ###======================================================================================================
    ### TransitionUp::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)
    #end

    ###======================================================================================================
    ### TransitionUp::forward()
    ###======================================================================================================
    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True)

        if concat:                            
            out = torch.cat([out, skip], 1)
        #end
          
        return out
    #end
#end


########################################################################################################################
### A1. MyUpSampling (originated from rpnet)
########################################################################################################################
class MyUpSampling(nn.Module):
    ###======================================================================================================
    ### MyUpSampling::__init__()
    ###======================================================================================================
    def __init__(self):
        super().__init__()
    # end

    ###======================================================================================================
    ### MyUpSampling::forward()
    ###======================================================================================================
    def forward(self, x, h_new, w_new):
        out = F.interpolate(
            x,
            size=(h_new, w_new),
            mode="bilinear",
            align_corners=True)

        return out
    # end
# end


########################################################################################################################
### A2. MyDecoder (originated from rpnet)
########################################################################################################################
class MyDecoder(nn.Sequential):
    ###======================================================================================================
    ### MyDecoder::__init__()
    ###======================================================================================================
    def __init__(self, out_channels):
        super().__init__()
        ###
        self.add_module('conv', nn.Conv2d(in_channels=256, out_channels=256,
                                          kernel_size=(3, 3), stride=1, padding=1, bias=False))
        #self.add_module('norm', nn.Sequential())
        self.add_module('relu', nn.ReLU(inplace=True))

        ###
        self.add_module('conv_b', nn.Conv2d(in_channels=256, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=1, padding=0, bias=False))

    ###======================================================================================================
    ### MyDecoder::forward()
    ###======================================================================================================
    def forward(self, x):
        return super().forward(x)
    #end
#end


"""
class ConvLayer(nn.Sequential):
    ###======================================================================================================
    ### ConvLayer::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
    #end


    ###======================================================================================================
    ### ConvLayer::forward()
    ###======================================================================================================
    def forward(self, x):
        return super().forward(x)
    #end
#end
"""

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
### 5. rpnet_c
########################################################################################################################
class rpnet_c(nn.Module):
    ############################################################################################################
    ### rpnet_c::__init__()
    ############################################################################################################
    def __init__(self, n_classes_ins = 16, n_classes_seg = 3):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(rpnet_c, self).__init__()

        ###================================================================================================
        ### parameters (for FC-HarDNet)
        ###================================================================================================

        ### output ch of init-conv
        first_ch = [16, 24, 32, 48]

        ### output of [HDB-xD & TDx], where HDB-xD is HDB-xDown, TDx is Transition Down-x
        ch_list  = [64, 96, 160, 224, 320]

        ###
        grmul    = 1.7

        ### output of first bottleneck layer in each [HDB-xD & TDx]
        gr       = [10, 16, 18, 24, 32]

        ### the number of bottleneck layers in each [HDB-xD & TDx]
        n_layers = [4, 4, 8, 8, 8]

        ###
        blks = len(n_layers)            # => blks: 5


        ###================================================================================================
        ### parameters (for rpnet)
        ###================================================================================================
        ch_output_fc_hardnet_a = 48         # output of HDB-3U
        ch_output_fc_hardnet_b = n_classes_seg         # output of Conv-Final


        ###================================================================================================
        ### init modules (for rpnet)
        ###================================================================================================
        #self.rpnet_decoder_cnvs = None
        self.rpnet_decoder_hmap_center = None


        ###================================================================================================
        ### init modules (for FC-HarDNet)
        ###================================================================================================
        self.base = nn.ModuleList([])


        ###================================================================================================
        ### set init-conv (for FC-HarDNet)
        ###================================================================================================
        self.base.append( ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, dilation_this=1) )  # conv0
        self.base.append( ConvLayer(first_ch[0], first_ch[1],  kernel=3, dilation_this=1) )                          # conv1
        self.base.append( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2, dilation_this=1) )                # conv2
        self.base.append( ConvLayer(first_ch[2], first_ch[3],  kernel=3, dilation_this=1) )                          # conv3
            # created
            #   (conv0): Conv2d( 3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #   (conv1): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            #   (conv2): Conv2d(24, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #   (conv3): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


        ###================================================================================================
        ### down-network
        ###================================================================================================
        self.shortcut_layers           = []     # self.shortcut_layers          = [4, 7, 10, 13]
        self.set_idx_module_HarDBlock  = []     # self.set_idx_module_HarDBlock = [4, 7, 10, 13, 16]
        skip_connection_channel_counts = []

        ###
        ch = first_ch[3]                        # output ch of last conv in init-conv

        ###
        for i_blk in range(blks):
            ###///////////////////////////////////////////////////////////////////////////////////////
            ### <<FC-HarDNet - base>>
            ###///////////////////////////////////////////////////////////////////////////////////////

            ###------------------------------------------------------------------------------
            ### create HarDBlock
            ###------------------------------------------------------------------------------
            #print('idx_blk:[%d], ch_in:[%d]' % (i,ch))
            if i_blk == blks-1:
                blk = HarDBlock(ch, gr[i_blk], grmul, n_layers[i_blk], have_dilation=True)
            else:
                blk = HarDBlock(ch, gr[i_blk], grmul, n_layers[i_blk])


            ch = blk.get_out_ch()
            ch_out_HarDBlock_this = blk.get_out_ch()
            #print('idx_blk:[%d], ch_out:[%d]' % (i,ch))


            ###------------------------------------------------------------------------------
            ### append HarDBlock & create shortcut
            ###------------------------------------------------------------------------------

            ### append ch into skip_connection_channel_counts
            skip_connection_channel_counts.append(ch)

            ### append HarDBlock
            self.base.append( blk )                                               # APPEND

            ### append set_idx_module_HarDBlock
            self.set_idx_module_HarDBlock.append( len(self.base)-1 )              # APPEND

            ### append module index (in ModuleList) into self.shortcut_layers
            if i_blk < (blks-1):
                self.shortcut_layers.append( len(self.base)-1 )
            #end


            ###------------------------------------------------------------------------------
            ### create transition-down layer
            ###------------------------------------------------------------------------------
            #print('idx_blk:[%d], conv for ch_in:[%d]' % (i_blk,ch))
            self.base.append( ConvLayer(ch, ch_list[i_blk], kernel=1) )           # APPEND
            ch = ch_list[i_blk]
            #print('idx_blk:[%d], conv for ch_out:[%d]' % (i_blk,ch))


            ###------------------------------------------------------------------------------
            ### create AvgPool2d
            ###------------------------------------------------------------------------------
            if i_blk < blks-1:
                self.base.append( nn.AvgPool2d(kernel_size=2, stride=2) )         # APPEND
            #end


            ###///////////////////////////////////////////////////////////////////////////////////////
            ### <<RPNet - backbone>>
            ###///////////////////////////////////////////////////////////////////////////////////////
            # print('idx_blk:[%d], ch_out_HarDBlock_this:[%d]' % (i_blk, ch_out_HarDBlock_this))
                # idx_blk: [0], ch_out_HarDBlock_this: [48]
                # idx_blk: [1], ch_out_HarDBlock_this: [78]
                # idx_blk: [2], ch_out_HarDBlock_this: [160]
                # idx_blk: [3], ch_out_HarDBlock_this: [214]
                # idx_blk: [4], ch_out_HarDBlock_this: [286]
        #end
        # completed to set
        #       <<for FC-HarDNet>>
        #         self.base[]
        #         self.shortcut_layers[]: having module index (in ModuleList) for shortcut (= here, skip_connection)
        #         skip_connection_channel_counts[]: having output ch of each HDB block (not transition-down layer)
        #       <<for RPNet-backbone>>
        #         self.set_idx_module_HarDBlock[]
        # ---------------------------------------------------------------------------------------------------------
        # note that
        #       skip_connection_channel_counts[] is for creating modules in the below (inside __init__()).
        #       self.shortcut_layers[] is used in forward().
        #---------------------------------------------------------------------------------------------------------
        # created HarDBlock & ConvL having
        #       idx_blk:[0], HarDBlock<ch_in: [48], ch_out: [48]>, ConvL<ch_in: [48], ch_out: [64]>
        #       idx_blk:[1], HarDBlock<ch_in: [64], ch_out: [78]>, ConvL<ch_in: [78], ch_out: [96]>
        #       idx_blk:[2], HarDBlock<ch_in: [96], ch_out: [160]>, ConvL<ch_in: [160], ch_out: [160]>
        #       idx_blk:[3], HarDBlock<ch_in: [160], ch_out: [214]>, ConvL<ch_in: [214], ch_out: [224]>
        #       idx_blk:[4], HarDBlock<ch_in: [224], ch_out: [286]>, ConvL<ch_in: [286], ch_out: [320]>
        #---------------------------------------------------------------------------------------------------------
        # To see the whole structure,
        #       please see FC_hardnet at /home/yu1/Desktop/avin_seg/seg_hardnet/
        #---------------------------------------------------------------------------------------------------------
        # message (2020/5/25)
        #   please, check shortcut
        #---------------------------------------------------------------------------------------------------------


        ###================================================================================================
        ### up-network
        ###================================================================================================
        cur_channels_count  = ch            # cur_channels_count: 320
        prev_block_channels = ch            # prev_block_channels: 320
        n_blocks            = blks-1        # n_blocks: 4, blks: 5
        self.n_blocks       = n_blocks

        ###
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])

        ###
        for i in range(n_blocks-1,-1,-1):
            #print('-'*50)
            #print('i:[%d]' % i)

            ###------------------------------------------------------------------------------
            ### append transition-up in self.transUpBlocks[]
            ###------------------------------------------------------------------------------
            self.transUpBlocks.append( TransitionUp(prev_block_channels, prev_block_channels) )         # APPEND
            #print('  prev_block_channels:[%d]' % prev_block_channels)


            ###------------------------------------------------------------------------------
            ### append conv1x1 in self.conv1x1_up[]
            ###------------------------------------------------------------------------------
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append( ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1) )    # APPEND
            #print('  cur_channels_count:[%d], skip_connection_channel_counts:[%d]' % (cur_channels_count, skip_connection_channel_counts[i]))
            #print('  cur_channels_count//2:[%d]' % (cur_channels_count//2))


            ###------------------------------------------------------------------------------
            ### append HarDBlock in self.denseBlocksUp[]
            ###------------------------------------------------------------------------------
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            self.denseBlocksUp.append(blk)                                      # APPEND


            ###------------------------------------------------------------------------------
            ### shift
            ###------------------------------------------------------------------------------
            prev_block_channels = blk.get_out_ch()
            #print('  [in] cur_channels_count: [%d]' % cur_channels_count)
            #print('  [out] blk.get_out_ch():[%d]' % prev_block_channels)

            cur_channels_count = prev_block_channels
        #end


        ###================================================================================================
        ### final conv (for FC-HarDNet)
        ###================================================================================================
        self.finalConvIns = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes_ins, kernel_size=1, stride=1,
                                   padding=0, bias=True)

        self.finalConvSeg = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes_seg, kernel_size=1, stride=1,
                                   padding=0, bias=True)


        ###================================================================================================
        ### relu for outcome of final conv (for rpnet)
        ###================================================================================================
        self.relu_on_finalConvIns = nn.ReLU(inplace=True)
        self.relu_on_finalConvSeg = nn.ReLU(inplace=True)


        ###================================================================================================
        ### rpnet_decoder_cnvs (for rpnet)
        ###================================================================================================
        #ch_in_rpnet_decoder_cnvs = ch_output_fc_hardnet_a + ch_output_fc_hardnet_b
        #self.rpnet_decoder_cnvs = ConvLayer(in_channels=ch_in_rpnet_decoder_cnvs, out_channels=256, kernel=3)


        ###================================================================================================
        ### rpnet_decoder_hmap_center (for rpnet)
        ###================================================================================================
        ch_in_rpnet_decoder_centerline = ch_output_fc_hardnet_a + ch_output_fc_hardnet_b


        self.rpnet_decoder_centerline = nn.Conv2d(in_channels=ch_in_rpnet_decoder_centerline,
                                                  out_channels=1, kernel_size=1, stride=1,
                                                  padding=0, bias=True)

        # self.rpnet_decoder_centerline_one = nn.Conv2d(in_channels=8,
        #                                           out_channels=1, kernel_size=1, stride=1,
        #                                           padding=0, bias=True)

        # self.rpnet_decoder_leftright  = nn.Conv2d(in_channels=ch_in_rpnet_decoder_centerline,
        #                                           out_channels=2, kernel_size=1, stride=1,
        #                                           padding=0, bias=True)

        # self.rpnet_decoder_leftright_one = nn.Conv2d(in_channels=8,
        #                                           out_channels=2, kernel_size=1, stride=1,
        #                                           padding=0, bias=True)


    #end

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    ############################################################################################################
    ### hardnet::forward()
    ############################################################################################################
    def forward(self, x):

        ###================================================================================================
        ### init
        ###================================================================================================

        ###
        skip_connections = []
        size_in = x.size()


        ###================================================================================================
        ### down-network
        ###================================================================================================
        for idx_module in range(len(self.base)):
            ###------------------------------------------------------------------------------
            ###
            ###------------------------------------------------------------------------------
            x = self.base[idx_module](x)
            # print(x.shape)


            ###------------------------------------------------------------------------------
            ### (For FC-HarDNet)
            ###------------------------------------------------------------------------------
            if idx_module in self.shortcut_layers:
                skip_connections.append(x)
            #end
        #end


        ###================================================================================================
        ### up-network (for FC-HarDNet)
        ###================================================================================================
        out_seg = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out_seg = self.transUpBlocks[i](out_seg, skip, True)
            out_seg = self.conv1x1_up[i](out_seg)
            out_seg = self.denseBlocksUp[i](out_seg)
        #end


        ###================================================================================================
        ### insert semantic segmentation output (HDB-3U outcome) to backbone_rpnet
        ###================================================================================================
        backbone_rpnet = out_seg


        ###================================================================================================
        ### [FC-HarDNet] final conv
        ###================================================================================================
        out_ins  = self.finalConvIns(backbone_rpnet)
        out_segm = self.finalConvSeg(backbone_rpnet)


        ###================================================================================================
        ### [rpnet] insert semantic segmentation output (final conv outcome) to backbone_rpnet
        ###================================================================================================
        out_seg_after_relu = self.relu_on_finalConvSeg(out_segm)
        backbone_rpnet = torch.cat([backbone_rpnet, out_seg_after_relu], 1)



        ###================================================================================================
        ### [rpnet] rpnet_decoder_hmap_center
        ###================================================================================================
        out_centerline = self.rpnet_decoder_centerline(backbone_rpnet)
        # out_leftright  = self.rpnet_decoder_leftright(backbone_rpnet)



        ###================================================================================================
        ### [FC-HarDNet] interpolate for final result
        ###================================================================================================
        out_ins_final = F.interpolate(out_ins, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        out_seg_final = F.interpolate(out_segm, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

            # completed to set
            #       out_seg_final


        ###================================================================================================
        ### [rpnet] interpolate for final result
        ###================================================================================================
        out_centerline_final = F.interpolate(out_centerline, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        # out_leftright_final  = F.interpolate(out_leftright, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

            # completed to set
            #       out_labelmap_left_right_final


        # h = out_centerline_final.shape[2]
        # w = out_centerline_final.shape[3]
        # out_centerline_for_regu = out_centerline_final.clone()
        # out_centerline_for_regu = out_centerline_for_regu * 0.
        # for img_index in range(out_centerline_final.shape[0]):
        #     peaks,indices = F.max_pool1d(out_centerline_final[img_index], kernel_size = 10 , stride=10, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        #     for row_counter in range(200,indices.shape[1]):
        #         for col_counter in range(indices.shape[2]):
        #             if indices[0,row_counter,col_counter] % 10 != 0 and indices[0,row_counter,col_counter] % 10 != 9 and peaks[0,row_counter,col_counter] > 10:
        #                 rail_width = peaks[0,row_counter,col_counter]
        #                 out_centerline_for_regu[img_index, 0, row_counter, min(w-1,indices[0, row_counter, col_counter]+rail_width.long()) ] = 1.
        #                 out_centerline_for_regu[img_index, 0, row_counter, max(0,indices[0, row_counter, col_counter]-rail_width.long()) ] = 1.

        return out_ins_final, out_seg_final, out_centerline_final
    #end


########################################################################################################################
### main()
########################################################################################################################
# see also
#   /home/yu1/PycharmProjects_b/proj_centernet_ut_austin/nets/hourglass_rpnet.py
if __name__ == '__main__':
    ###
    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
    #end


    ###
    net = rpnet_c(n_classes_ins=16, n_classes_seg=3).cuda()


    ###
    with torch.no_grad():
        out_seg, out_centerline, out_leftright = net(torch.randn(2, 3, 540, 960).cuda())
        #y = net(torch.randn(2, 3, 512, 512).cuda())
        #y = net(torch.randn(2, 3, 1024, 512).cuda())
        print("out_seg.size()")
        print(out_seg.size())
        print("out_centerline.size()")
        print(out_centerline.size())
        print("out_labelmap_leftright.size()")
        print(out_leftright.size())
    #end


#end

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


#out = torch.cat([out, skip], 1)
#size = (skip.size(2), skip.size(3)),
#backbone_rpnet = torch.cat([backbone_rpnet, x_rpnet], 1)


