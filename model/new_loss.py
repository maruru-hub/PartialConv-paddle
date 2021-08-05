
import paddle
import paddle.nn as nn
from paddle.vision import models
class VGG16(nn.Layer):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = nn.Sequential()
        self.relu1_2 = nn.Sequential()

        self.relu2_1 = nn.Sequential()
        self.relu2_2 = nn.Sequential()

        self.relu3_1 = nn.Sequential()
        self.relu3_2 = nn.Sequential()
        self.relu3_3 = nn.Sequential()
        self.max3 = nn.Sequential()


        self.relu4_1 = nn.Sequential()
        self.relu4_2 = nn.Sequential()
        self.relu4_3 = nn.Sequential()


        self.relu5_1 = nn.Sequential()
        self.relu5_2 = nn.Sequential()
        self.relu5_3 = nn.Sequential()

        for x in range(2):
            self.relu1_1.add_sublayer(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_sublayer(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_sublayer(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_sublayer(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_sublayer(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_sublayer(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_sublayer(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_sublayer(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_sublayer(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_sublayer(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_sublayer(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_sublayer(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_sublayer(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_sublayer(str(x), features[x])


        # don't need the gradients, just want the features
        for param in self.parameters():
            param.stop_gradient = True

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)


        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)


        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3':max_3,


            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,


            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out
class StyleLoss(nn.Layer):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self,vgg):
        super(StyleLoss, self).__init__()
        self.vgg=vgg
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        print(x_vgg['relu3_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss
class PerceptualLoss(nn.Layer):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_sublayer('vgg', VGG16())
        self.criterion = nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



# class DESTLOSS(nn.Module):
#     def __init__(self):
#         super(DESTLOSS, self).__init__()
#         self.criterion = torch.nn.L1Loss()
#
#     def __call__(self, Gt_de, Gt_st, Fake_de, Fake_st):
#         Gt_de = F.interpolate (Gt_de, size=(32,32), mode='bilinear')
#         Gt_st = F.interpolate (Gt_st, size=(32,32), mode='bilinear')
#
#
#
#         return content_loss
if __name__=='__main__':
    # P=PerceptualLoss()
    S=StyleLoss(VGG16())
    a = paddle.randn([1,3,256,256])
    b = paddle.randn([1,3,256,256])
    print(a)
    print(b)
    # ploss=P(a,b)
    sloss=S(a,b)
    # print(ploss)
    print(sloss)