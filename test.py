from options import OPT
from model.base_model import BASE
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
    d_statedict_model = paddle.load(opt.checkpoints_dir + "model/en.pdparams")
    model.net_EN.set_state_dict(d_statedict_model)

    g_statedict_model = paddle.load(opt.checkpoints_dir + "model/de.pdparams")
    model.net_DE.set_state_dict(g_statedict_model)

    mask_root = './test_data/irregular_mask/30-40/06174'
    de_root = './test_data/celeba_hq'
    results_dir = r'./result/30-40/06174'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    mask_paths = glob('{:s}/*'.format(mask_root))
    de_paths = glob('{:s}/*'.format(de_root))
    image_len = len(de_paths )

    for i in tqdm(range(image_len)):
        #可以用random的方法读取随机的一张mask
        path_m = mask_paths[0]
        path_d = de_paths[i]

        s = re.findall(r'\d+\.?\d*',path_d)

        mask = Image.open(path_m).convert("RGB")
        detail = Image.open(path_d).convert("RGB")

        mask = mask_transform(mask)
        detail = img_transform(detail)
        mask = paddle.unsqueeze(mask, 0)

        detail = paddle.unsqueeze(detail, 0)

        with paddle.no_grad():
            model.set_input(detail, mask)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask + detail*(1-mask)
            fake_image = (fake_out+1)/2.0
        output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(rf"{results_dir}/{s[0]}.png")



