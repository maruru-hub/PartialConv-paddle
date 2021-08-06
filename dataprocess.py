
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as transforms
import random
from glob import glob
from PIL import Image
from options import OPT

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

        self.opt = opt
        self.img_root = sorted(glob('{:s}/*.jpg'.format(self.opt.dataroot)))
        self.mask_root = sorted(glob('{:s}/*.png'.format(self.opt.maskroot)))
        self.N_mask = len(self.mask_root)
        print(len(self.img_root))
        print(self.N_mask)

    def __getitem__(self, idx):
        img = Image.open(self.img_root[idx])
        img = img.convert('RGB')
        mask = Image.open(self.mask_root[random.randint(0, self.N_mask - 1)])
        mask = mask.convert('RGB')
        de_img = self.img_transform(img)
        de_mask = self.mask_transform(mask)

        return de_img, de_mask

    def __len__(self):
        return len(self.img_root)



