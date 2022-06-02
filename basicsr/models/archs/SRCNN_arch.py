# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from torchsummary import summary


class SRCNNBlock(nn.Module):
    def __init__(self, c, c_expand=128, kernel_sizes=[], strides=[]):
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        dw_channel = c * c_expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=kernel_sizes[0], 
                                padding=0, stride=strides[0], groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=kernel_sizes[1], 
                                padding=0, stride=strides[1], groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=kernel_sizes[2], 
                        padding=0, stride=strides[2], groups=1, bias=True)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 4, kernel_size=kernel_sizes[2], 
        #                         padding=0, stride=strides[2], groups=1, bias=True)
        # self.conv4 = nn.Conv2d(in_channels=dw_channel // 4, out_channels=c, kernel_size=kernel_sizes[3], 
        #                         padding=0, stride=strides[3], groups=1, bias=True)

    def forward(self, inp):
        x = inp

        x = self.CircularPadding(x, 0)
        x = self.conv1(x)
        x = F.relu(x)

        x = self.CircularPadding(x, 1)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.CircularPadding(x, 2)
        x = self.conv3(x)

        # x = self.CircularPadding(x, 2)
        # x = self.conv3(x)
        # x = F.relu(x)

        # x = self.CircularPadding(x, 3)
        # x = self.conv4(x)

        return x

    def CircularPadding(self, inp, iconv):
        _, _, H, W = inp.shape
        kht, kwd = self.kernel_sizes[iconv]
        sht, swd = self.strides[iconv]
        assert kwd%2 != 0 and kht%2 !=0 and (W-kwd)%swd==0 and (H-kht)%sht ==0, 'kernel_size should be odd, (dim-kernel_size) should be divisible by stride'

        pwd = int((W - 1 - (W - kwd) / swd) // 2)
        pht = int((H - 1 - (H - kht) / sht) // 2)
        
        # kht1, kwd1 = self.kernel_sizes[1]
        # kht2, kwd2 = self.kernel_sizes[2]
        # pwd = int((W - 1 - (W - kwd) / swd) // 2 + (W - 1 - (W - kwd1) / swd) // 2 + (W - 1 - (W - kwd2) / swd) // 2)
        # pht = int((H - 1 - (H - kht) / sht) // 2 + (H - 1 - (H - kht1) / sht) // 2 + (H - 1 - (H - kht2) / sht) // 2)
        
        x = F.pad(inp, (pwd, pwd, pht, pht), 'circular')

        return x


class SRCNN(nn.Module):

    def __init__(self, img_channel=1, c_expand=64, kernel_sizes=[], strides=[]):
        super().__init__()

        self.blk = SRCNNBlock(img_channel, c_expand, kernel_sizes, strides)

    def forward(self, inp):
        return self.blk(inp)

    # def __init__(self, img_channel=1, c_expand=64, kernel_sizes=[], strides=[]):
    #     super().__init__()

    #     self.layer1 = torch.nn.Sequential(
    #         torch.nn.Conv2d(1, 64, kernel_size=(21,9), stride=1, padding=0, bias=True),
    #         torch.nn.ReLU())

    #     self.layer2 = torch.nn.Sequential(
    #         torch.nn.Conv2d(64, 32, kernel_size=(15,15), stride=1, padding=0, bias=True),
    #         torch.nn.ReLU())

    #     self.layer3 = torch.nn.Sequential(
    #         torch.nn.Conv2d(32, 1, kernel_size=(5,5), stride=1, padding=0, bias=True))

    # def forward(self, x):
    #     W = 240
    #     H = 960
    #     kwd = 9
    #     kht = 21
    #     kwd1 = 15
    #     kht1 = 15
    #     kwd2 = 5
    #     kht2 = 5
    #     swd = 1
    #     sht = 1
    #     pwd = int((W - 1 - (W - kwd) / swd) // 2 + (W - 1 - (W - kwd1) / swd) // 2 + (W - 1 - (W - kwd2) / swd) // 2)
    #     pht = int((H - 1 - (H - kht) / sht) // 2 + (H - 1 - (H - kht1) / sht) // 2 + (H - 1 - (H - kht2) / sht) // 2)
        
    #     x = F.pad(x, (pwd, pwd, pht, pht), 'circular')
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     return out


if __name__ == '__main__':
    import resource
    def using(point=""):
        # print(f'using .. {point}')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        global Total, LastMem

        # if usage[2]/1024.0 - LastMem > 0.01:
        # print(point, usage[2]/1024.0)
        print(point, usage[2] / 1024.0)

        LastMem = usage[2] / 1024.0
        return usage[2] / 1024.0

    img_channel = 1
    img_ht = 960
    img_wd = 240
    strides = [(3, 1), (1, 1), (1, 1)]
    kernel_sizes = [(15, 9), (3, 3), (5, 5)]
    c_expand = 128

    print('kernel_sizes', kernel_sizes, 'strides', strides, 'c_expand', c_expand)
    
    using('start . ')
    net = SRCNNNet(img_channel=img_channel, c_expand=c_expand, kernel_sizes=kernel_sizes, strides=strides)

    using('network .. ')

    # for n, p in net.named_parameters()
    #     print(n, p.shape)

    inp = torch.randn((4, img_channel, img_ht, img_wd))

    out = net(inp)
    final_mem = using('end .. ')
    # out.sum().backward()

    # out.sum().backward()

    # using('backward .. ')

    # exit(0)

    # keras like model summary
    summary(net, input_size=(img_channel, img_ht, img_wd), device='cpu')

    # inp_shape = (3, 512, 512)

    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    # print(macs, params)

    # print('total .. ', params * 8 + final_mem)



