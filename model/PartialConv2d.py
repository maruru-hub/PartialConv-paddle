import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision import models
import functools


class VGG16FeatureExtractor(nn.Layer):
    def __init__(self):
        super(VGG16FeatureExtractor,self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])  # torch.size([5,h,w])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():  # getattr()用于返回一个对象属性值
                param.requires_grad = False  # 不需要保存梯度

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PartialConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=None):
        super().__init__()
        self.input_conv = nn.Conv2D(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.mask_conv = nn.Conv2D(1, 1, kernel_size,
                                   stride, padding, dilation, groups, bias_attr=False,
                                   weight_attr=nn.initializer.Constant(value=1.0))

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        #mask:black=1,white=0
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        input1=input*mask
        output = self.input_conv(input1)  ##原来是input*mask，我改了

        # if self.input_conv.bias is not None:
        #     output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
        #         output)
        # else:
        #     output_bias = paddle.zeros_like(output)

        with paddle.no_grad():
            output_mask = self.mask_conv(mask)

        new_mask = paddle.clip(output_mask, 0, 1)
        # no_update_holes = output_mask == 0
        # mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        # output_pre = (output - output_bias) / mask_sum + output_bias
        # output = output_pre.masked_fill_(no_update_holes, 0.0)

        # new_mask = paddle.ones_like(output)
        # new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        return output, new_mask


class PCBActiv(nn.Layer):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=None):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)
        if bn:
            self.bn = bn
        self.norm_layer = nn.BatchNorm2D(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)

        if hasattr(self, 'bn'):
            h = self.norm_layer(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


if __name__ == '__main__':
    Acb = PCBActiv(3, 64, bn=False, activ='', sample='down-7')
    a = paddle.randn([1, 3, 9, 9])
    b = paddle.ones([1, 1, 9, 9])
    a, b = Acb(a, b)
    print(a,b)
