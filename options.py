#添加设置全局参数

import warnings
warnings.filterwarnings('ignore')

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# set up global parameters
# 修改了syncbatchnorm为batchnorm
# 修改了 dataroot
# batchSize 设为 4
# 调整 vgg loss lambda 为 0.2
# 优化器的参数和学习率直接在下面，不用OPT设置。
d_lr, g_lr, beta1, beta2 = 4e-4, 1e-4, 0., .999

class OPT():
    def __init__(self):
        super(OPT, self).__init__()
        # self.batchSize=1
        self.batchSize=1
        self.beta1=0.5
        self.beta2=0.999
        self.cache_filelist_read=True
        self.cache_filelist_write=True
        self.checkpoints_dir='./checkpoints'
        self.coco_no_portraits=False
        self.contain_dontcare_label=True
        self.continue_train=False
        self.crop_size=256
        # self.dataroot='./datasets/cityscapes/'
        self.maskroot='./datasets/irregular_mask'
        self.dataroot='./datasets/celeba'
        # self.dataroot='/home/aistudio/coco_stuff/'
        self.dataset_mode='inpainting'
        self.display_freq=500
        self.display_winsize=256
        self.gan_mode='hinge'
        self.gpu_ids=[]
        self.init_type='xavier'
        self.init_variance=0.02
        self.isTrain=True
        self.label_nc=182
        self.lambda_feat=10.0
        self.lambda_kld=0.05
        # self.lambda_vgg=10.0
        self.lambda_vgg=.2
        self.load_from_opt_file=False
        self.load_size=256
        self.d_lr=0.0002
        self.g_lr=0.0002
        self.model='pix2pix'
        self.nThreads=0
        self.n_layers_D=4
        self.name='Celeba'
        self.ndf=64
        self.nef=16
        self.netD='multiscale'
        self.netD_subarch='n_layer'
        self.netG='spade'
        self.ngf=64
        self.niter=50
        self.niter_decay=0
        self.no_ganFeat_loss=False
        self.no_html=False
        self.no_instance=False
        self.no_pairing_check=False
        self.no_vgg_loss=False
        self.norm_D='spectralinstance'
        self.norm_E='spectralinstance'
        # self.norm_G='spectralspadesyncbatch3x3'
        self.norm_G='spectralspadebatch3x3'
        self.num_D=2
        self.num_upsampling_layers='normal'
        self.optimizer='adam'
        self.output_nc=3
        self.phase='train'
        self.preprocess_mode='resize_and_crop'
        self.print_freq=100
        self.save_epoch_freq=5
        self.save_latest_freq=5000
        self.which_epoch='latest'
        self.num_workers=4
        self.current_epoch=0
        self.epoch_num=2
        self.lambda_GAN=0.2
        self.lambda_style=120
        self.lambda_prc=0.05
        self.lambda_hole=6
        self.lambda_valid=1
        self.lambda_l1=1
        self.lambda_tv=0.1

        self.log_dir='./logs'
opt = OPT()