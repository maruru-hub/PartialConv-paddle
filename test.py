import time
import pdb
from options import OPT
from dataprocess import InpaintDateset
from model.base_model import BASE
import paddle.vision
#from torch.utils.tensorboard import SummaryWriter
import os
import paddle
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import re
import paddle.vision.transforms as transforms

if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    opt = OPT()
    model = BASE(opt)
    print('读取存储的模型权重、优化器参数...')
    d_statedict_model = paddle.load(opt.checkpoints_dir + "model/21_en.pdparams")
    model.net_EN.set_state_dict(d_statedict_model)

    g_statedict_model = paddle.load(opt.checkpoints_dir + "model/21_de.pdparams")
    model.net_DE.set_state_dict(g_statedict_model)
    # en = torch.load("checkpoints/celeba-irregular2/65_net_EN.pth")
    # de = torch.load("checkpoints/celeba-irregular2/65_net_DE.pth")
    #85

    #预训练模型的写法
    # model.netEN.module.load_state_dict(torch.load("EN.pkl"))
    # model.netDE.module.load_state_dict(torch.load("DE.pkl"))
    # model.netMEDFE.module.load_state_dict(torch.load("MEDEF.pkl"))
    mask_root = './test_data/irregular_mask/30-40/06174'
    de_root = './test_data/celeba_hq'
    results_dir = r'./result/30-40/06174'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    mask_paths = glob('{:s}/*'.format(mask_root))
    de_paths = glob('{:s}/*'.format(de_root))
    # st_path = glob('{:s}/*'.format(opt.st_root))
    image_len = len(de_paths )

    for i in tqdm(range(image_len)):
        # only use one mask for all image
        path_m = mask_paths[0]
        path_d = de_paths[i]
        # path_s = de_paths[i]

        s = re.findall(r'\d+\.?\d*',path_d)

        mask = Image.open(path_m).convert("RGB")
        detail = Image.open(path_d).convert("RGB")
        # structure = Image.open(path_s).convert("RGB")

        mask1 = mask_transform(mask)
        detail1 = img_transform(detail)
        # structure = img_transform(structure)
        mask1 = paddle.unsqueeze(mask1, 0)

        detail1 = paddle.unsqueeze(detail1, 0)
        # structure = torch.unsqueeze(structure,0)

        with paddle.no_grad():
            model.set_input(detail1, mask1)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask1 + detail1*(1-mask1)
            fake_image = (fake_out+1)/2.0
        output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(rf"{results_dir}/{s[0]}.png")

        # print('{}/{}'.format(i,image_len))


