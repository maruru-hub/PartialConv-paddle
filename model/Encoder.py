import paddle.nn as nn
from model.PartialConv2d import PCBActiv


# 定义encoder

# define the encoder skip connect
class UnetSkipConnectionEBlock(nn.Layer):
    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False, use_dropout=False):
        super(UnetSkipConnectionEBlock,self).__init__()
        downconv = nn.Conv2D(outer_nc,inner_nc,kernel_size=4,stride=2,padding=1)
        conv = nn.Conv2D(outer_nc,inner_nc,kernel_size=5,stride=1,padding=2)
        downrelu = nn.LeakyReLU(0.2,True)
        downnorm = nn.BatchNorm2D(inner_nc)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downconv, downrelu]
            model = down
        else:
            down = [downconv, downrelu, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# 定义残差block
class ResnetBlock(nn.Layer):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Pad2D(dilation, mode='reflect'),
            nn.Conv2D(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias_attr=False),
            nn.InstanceNorm2D(dim),
            nn.ReLU(True),
            nn.Pad2D(1, mode='reflect'),
            nn.Conv2D(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias_attr=False),
            nn.InstanceNorm2D(dim),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, res_num=4, use_dropout=False):
        super(Encoder, self).__init__()

        # Unet structure
        self.ec_1 = PCBActiv(input_nc, ngf, bn=False, activ=None, sample='down-7')
        self.ec_2 = PCBActiv(ngf, ngf * 2,sample='down-5')
        self.ec_3 = PCBActiv(ngf * 2, ngf * 4, sample='down-5')
        self.ec_4 = PCBActiv(ngf * 4, ngf * 8, sample='down-3')
        self.ec_5 = PCBActiv(ngf * 8, ngf * 8, sample='down-3')
        self.ec_6 = PCBActiv(ngf * 8, ngf * 8, bn=False, sample='down-3')

        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

    def forward(self, input, mask):
        y_1, m_1 = self.ec_1(input, mask)
        y_2, m_2 = self.ec_2(y_1, m_1)
        y_3, m_3 = self.ec_3(y_2, m_2)
        y_4, m_4 = self.ec_4(y_3, m_3)
        y_5, m_5 = self.ec_5(y_4, m_4)
        y_6, _ = self.ec_6(y_5, m_5)
        y_7 = self.middle(y_6)
        return y_1, y_2, y_3, y_4, y_5, y_7

