
from paddle.io import Dataset, DataLoader
# from paddle.vision.transforms import Resize
import paddle.vision.transforms as transforms
import numpy as np
import random
from glob import glob
from PIL import Image
from options import OPT

# 加载数据集
# 处理图片数据：裁切、水平翻转、调整图片数据形状、归一化数据

    # self.maskroot='data/irregular_mask/irregular_mask'
    # self.dataroot='data/celeba_hq/celeba_hq/train'


# 定义Inpaint数据集对象
class InpaintDateset(Dataset):
    def __init__(self, opt):
        super(InpaintDateset, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.ToTensor()
        ])

        # img_dir = opt.dataroot+'train_img/'
        # _, _, image_list = next(os.walk(img_dir))
        # self.image_list = np.sort(image_list)
        self.opt = opt
        self.img_root = sorted(glob('{:s}/*.jpg'.format(self.opt.dataroot)))
        self.mask_root = sorted(glob('{:s}/*.png'.format(self.opt.maskroot)))
        self.N_mask = len(self.mask_root)
        # inst_dir = opt.dataroot+'/'
        # inst_list = next(os.walk(inst_dir))
        # self.inst_list = np.sort(inst_list)
        print(len(self.img_root))
        print(self.N_mask)

    def __getitem__(self, idx):
        img = Image.open(self.img_root[idx])
        img = img.convert('RGB')
        mask = Image.open(self.mask_root[random.randint(0, self.N_mask - 1)])
        mask = mask.convert('RGB')
        de_img = self.img_transform(img)
        de_mask = self.mask_transform(mask)

        # de_img = data_transform(img, load_size=opt.load_size, is_image=True)
        # de_mask = data_transform(mask, load_size=opt.load_size, is_image=False)

        # 把图片改成masked image

        return de_img, de_mask

    def __len__(self):
        return len(self.img_root)

if __name__=='__main__':
    opt = OPT()
    dataset = InpaintDateset(opt)

    # dataloader = DataLoader(dataset, batch_size=opt.load_size, shuffle=True, num_workers=2)

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True,
                        drop_last=True,
                        num_workers=4)

    for image,mask in loader:
        print(image,mask)


