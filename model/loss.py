# @Time    : 2021/8/4 22:12
# @Author  : maruru
#定义损失函数
import paddle
import paddle.nn as nn
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.shape
    # feat = feat.view(b, ch, h * w)
    feat = paddle.reshape(feat, [b, ch, h*w])
    feat_t = paddle.transpose(feat, perm=[0, 2, 1])
    gram = paddle.bmm(feat, feat_t) / (ch * h * w) #torch.bmm(a,b)，a和b进行矩阵的乘法，且a，b都是三维的tensor，必须满足a是(b,h,w)b是(b,w,h)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = paddle.mean(paddle.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        paddle.mean(paddle.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Layer):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output   #I_comp   mask是缺失区域为0

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:  #如果output的通道是三通道
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:  #torch.cat是将两个张量拼接在一起，cat是concatnate的意思，后面的数字代表是第几列开始拼接
            feat_output_comp = self.extractor(paddle.concat([output_comp]*3, 1))  #拼接第一列，相当于将b*1*h*w的tensor变成b*3*h*w
            feat_output = self.extractor(paddle.concat([output]*3, 1))
            feat_gt = self.extractor(paddle.concat([gt]*3, 1))
        else:
            raise ValueError('only gray an') #程序抛出异常

        loss_dict['prc'] = 0.0  #感知损失
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict

class GANLoss(nn.Layer):
    def __init__(self,  target_real_label=1.0, target_fake_label=0.0,
                 tensor=paddle.to_tensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

#这个我可以改，改成我自己定义的GAN
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                # self.real_label_var = self.Tensor(input.shape).fill_(self.real_label)
                self.real_label_var = paddle.full(input.shape,self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                # self.fake_label_var= self.Tensor(input.shape).fill_(self.real_label)
                self.fake_label_var= paddle.full(input.shape,self.real_label)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        if target_is_real:
            errD = (paddle.mean((y_pred - paddle.mean(y_pred_fake) - target_tensor) ** 2) + paddle.mean(
                (y_pred_fake - paddle.mean(y_pred) + target_tensor) ** 2)) / 2
            return errD
        else:
            errG = (paddle.mean((y_pred - paddle.mean(y_pred_fake) + target_tensor) ** 2) + paddle.mean(
                (y_pred_fake - paddle.mean(y_pred) - target_tensor) ** 2)) / 2
            return errG
