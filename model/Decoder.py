import paddle
import paddle.nn as nn
from model.Encoder import Encoder

# define the decoder skip connect
class UnetSkipConnectionDBlock(nn.Layer):
    def __init__(self, inner_nc, outer_nc, outermost=False, innermost=False, use_dropout=False):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upconv = nn.Conv2DTranspose(inner_nc,outer_nc,kernel_size=4,stride=2,padding=1)
        upnorm = nn.BatchNorm2D(outer_nc)
        if outermost:
            print('using relu,bn,conv')
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)

# define decoder
class Decoder(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False):
        super(Decoder, self).__init__()

        #Unet structure
        self.dc_1 = UnetSkipConnectionDBlock(ngf*8, ngf*8, use_dropout=use_dropout, innermost=True)
        self.dc_2 = UnetSkipConnectionDBlock(ngf*16, ngf*8, use_dropout=use_dropout)
        self.dc_3 = UnetSkipConnectionDBlock(ngf*16, ngf*4, use_dropout=use_dropout)
        self.dc_4 = UnetSkipConnectionDBlock(ngf*8, ngf*2, use_dropout=use_dropout)
        self.dc_5 = UnetSkipConnectionDBlock(ngf*4, ngf, use_dropout=use_dropout)
        self.dc_6 = UnetSkipConnectionDBlock(ngf*2, output_nc, use_dropout=use_dropout, outermost=True)

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.dc_1(input_6)
        y_2 = self.dc_2(paddle.concat([y_1, input_5], 1))
        y_3 = self.dc_3(paddle.concat([y_2, input_4], 1))
        y_4 = self.dc_4(paddle.concat([y_3, input_3], 1))
        y_5 = self.dc_5(paddle.concat([y_4, input_2], 1))
        y_6 = self.dc_6(paddle.concat([y_5, input_1], 1))
        out = y_6
        return out
