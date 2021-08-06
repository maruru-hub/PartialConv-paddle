import time
from dataprocess import InpaintDateset
import os
from model.base_model import BASE
import paddle
from paddle.io import Dataset, DataLoader
from visualdl import LogWriter
from options import OPT
from PIL import Image
import numpy as np

if __name__ == "__main__":

    opt = OPT()
    dataset = InpaintDateset(opt)
    loader = DataLoader(dataset,
                        batch_size=opt.batchSize,
                        shuffle=True,
                        drop_last=True,
                        num_workers=opt.num_workers)
    # define the dataset
    # Create model
    model = BASE(opt)
    total_steps = 0
    # Create the logs
    dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    if not os.path.exists(dir):
        os.mkdir(dir)

    # Start Training
    with LogWriter(logdir=dir) as writer:
        for epoch in range(1):
            epoch_start_time = time.time()
            epoch_iter = 0
            for detail, mask in loader():
                print("begin training")
                iter_start_time = time.time()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(detail, mask)
                model.optimize_parameters()
                # display the training processing 这部分可以注释掉
                if total_steps % opt.display_freq == 0:
                    input, output, GT = model.get_current_visuals()
                    input = input.detach().numpy()[0].transpose((1, 2, 0)) * 255
                    input = Image.fromarray(input.astype(np.uint8))
                    output = output.detach().numpy()[0].transpose((1, 2, 0)) * 255
                    output = Image.fromarray(output.astype(np.uint8))
                    GT = GT.detach().numpy()[0].transpose((1, 2, 0)) * 255
                    GT = Image.fromarray(GT.astype(np.uint8))
                    input.save(rf"./results/{epoch}_{total_steps}_input.png")
                    GT.save(rf"./results/{epoch}_{total_steps}_GT.png")
                    output.save(rf"./results/{epoch}_{total_steps}_output.png")
                #display the training loss  这部分可以注释掉
                if total_steps % opt.print_freq == 0:
                    errors = model.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    writer.add_scalar('G_GAN', value=errors['G_GAN'], step=total_steps + 1)
                    writer.add_scalar('G_L1', value=errors['G_L1'], step=total_steps + 1)
                    writer.add_scalar('D_loss', value=errors['D'], step=total_steps + 1)
                    writer.add_scalar('F_loss', value=errors['F'], step=total_steps + 1)
                    print('iters: %d iteration time: %.10fsec' % (total_steps, t))
