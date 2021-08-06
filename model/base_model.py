import paddle
import random
from collections import OrderedDict
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Discriminator import NLayerDiscriminator
from model.loss import InpaintingLoss,GANLoss
from model.PartialConv2d import VGG16FeatureExtractor
from options import OPT
import numpy as np

class BASE():
    def __init__(self, opt):
        super(BASE,self).__init__()

        self.opt = opt
        self.istrain=opt.isTrain
        self.criterion=InpaintingLoss(VGG16FeatureExtractor())
        self.criterionGAN = GANLoss()

        self.net_EN = Encoder(3,64)
        self.net_DE = Decoder(64,3)


        if self.istrain:
            self.netD=NLayerDiscriminator(3)
            self.netF=NLayerDiscriminator(3)
            self.net_EN.train()
            self.net_DE.train()
            self.netD.train()
            self.netF.train()
        self.opt_d = paddle.optimizer.Adam(learning_rate=opt.d_lr, beta1=opt.beta1, beta2=opt.beta2, parameters=self.netD.parameters())
        self.opt_f = paddle.optimizer.Adam(learning_rate=opt.d_lr, beta1=opt.beta1, beta2=opt.beta2, parameters=self.netF.parameters())
        self.opt_en = paddle.optimizer.Adam(learning_rate=opt.g_lr, beta1=opt.beta1, beta2=opt.beta2, parameters=self.net_EN.parameters())
        self.opt_de = paddle.optimizer.Adam(learning_rate=opt.g_lr, beta1=opt.beta1, beta2=opt.beta2, parameters=self.net_DE.parameters())


    def mask_process(self, mask):
        mask = mask[0][0]
        mask = paddle.unsqueeze(mask,0)
        mask = paddle.unsqueeze(mask,1)
        return mask

    def set_input(self, input_De, mask):

        self.Gt_DE = input_De
        self.input_DE = input_De
        self.mask_global = (1-self.mask_process(mask))#black=1,white=0
        self.Gt_Local = input_De
        # define local area which send to the local discriminator
        self.crop_x = random.randint(0, 191)
        self.crop_y = random.randint(0, 191)
        self.Gt_Local = self.Gt_Local[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        self.ex_mask = paddle.expand(self.mask_global,[self.mask_global.shape[0], 3, self.mask_global.shape[2],
                                               self.mask_global.shape[3]])


        # Do not set the mask regions as 0
        self.input_DE = self.input_DE*self.mask_global

    def forward(self):

        fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.net_EN(
            self.input_DE,self.mask_global)
        De_in = [fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6]
        self.fake_out = self.net_DE(De_in[0], De_in[1], De_in[2], De_in[3], De_in[4], De_in[5])

    def backward_D(self):
        fake_AB = self.fake_out   # image_G
        real_AB = self.Gt_DE  # GroundTruth
        real_local = self.Gt_Local
        fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global Discriminator
        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        # Local discriminator
        self.pred_fake_F = self.netF(fake_local.detach())
        self.pred_real_F = self.netF(real_local)
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F, self.pred_real_F, True)

        # Coarse discriminator

        self.loss_D = self.loss_D_fake + self.loss_F_fake
        self.loss_D.backward()

    def backward_G(self):
        # First, The generator should fake the discriminator
        real_AB = self.Gt_DE
        fake_AB = self.fake_out
        real_local = self.Gt_Local
        fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real = self.netD(real_AB)
        pred_fake = self.netD(fake_AB)
        # Local discriminator
        pred_real_F = self.netF(real_local)
        pred_fake_f = self.netF(fake_local)
        # coarse discriminator

        self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_F,
                                                                                             False)
        # Second, Reconstruction loss
        self.loss_dict=self.criterion(self.input_DE,self.mask_global,self.fake_out, self.Gt_DE)

        self.loss_G = self.loss_G_GAN * self.opt.lambda_GAN + self.loss_dict['style'] * self.opt.lambda_style + \
                      self.loss_dict['hole'] * self.opt.lambda_hole + self.loss_dict['valid'] * self.opt.lambda_valid + \
                      self.loss_dict['prc'] * self.opt.lambda_prc + self.loss_dict['tv'] * self.opt.lambda_tv

        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.stop_gradient = requires_grad
    def optimize_parameters(self):
        self.forward()

        # Optimize the D and F first
        for parma in self.net_EN.parameters():
            print(parma)
        self.set_requires_grad(self.net_EN, True)
        self.set_requires_grad(self.net_DE, True)
        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netD, False)

        self.opt_d.clear_grad()
        self.opt_f.clear_grad()

        self.backward_D()
        self.opt_f.step()
        self.opt_d.step()


        # Optimize EN, DE, MEDEF
        self.set_requires_grad(self.netF, True)
        self.set_requires_grad(self.netD, True)

        self.set_requires_grad(self.net_EN, False)
        self.set_requires_grad(self.net_DE, False)

        self.opt_en.clear_grad()
        self.opt_de.clear_grad()

        self.backward_G()
        self.opt_en.step()
        self.opt_de.step()

    def get_current_errors(self):
        print('Gan:%.4f,l1:%.4f,style:%.4f,tv:%.4f,prc:%.4f' % (
        self.loss_G_GAN.item(), self.loss_dict['hole']+self.loss_dict['valid'], self.loss_dict['style'], self.loss_dict['tv'],self.loss_dict['prc']))
        # show the current loss
        return OrderedDict([('G_GAN', self.loss_G_GAN),
                            ('G_L1', self.loss_G),
                            ('D', self.loss_D_fake),
                            ('F', self.loss_F_fake)
                            ])

    def get_current_visuals(self):
        input_image = (self.input_DE + 1) / 2.0
        fake_image = (self.fake_out + 1) / 2.0
        real_gt = (self.Gt_DE + 1) / 2.0

        return input_image[0].unsqueeze(0), fake_image[0].unsqueeze(0), real_gt[0].unsqueeze(0)

    def save_epoch(self,epoch):
        paddle.save(self.netD.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_d.pdparams")
        paddle.save(self.opt_d.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_d.pdopt")
        paddle.save(self.net_EN.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_en.pdparams")
        paddle.save(self.opt_en.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_en.pdopt")
        paddle.save(self.net_DE.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_de.pdparams")
        paddle.save(self.opt_de.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_de.pdopt")
        paddle.save(self.netF.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_f.pdparams")
        paddle.save(self.opt_f.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_f.pdopt")




