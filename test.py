import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
from tqdm import tqdm

import config
import myutils
import torchvision.utils as utils
import math
import torch.nn.functional as F

from torch.utils.data import DataLoader

##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"]='1'
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "ucf101":
    from dataset.ucf101_test import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "Davis":
    from dataset.Davis_test import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True)

if args.model == 'VFIT_S':
    from model.VFIT_S import UNet_3D_3D
elif args.model == 'VFIT_B':
    from model.VFIT_B import UNet_3D_3D

print("Building model: %s"%args.model)
model = UNet_3D_3D(n_inputs=args.nbr_frame, joinType=args.joinType)

model = torch.nn.DataParallel(model).to(device)
print("#params" , sum([p.numel() for p in model.parameters()]))

def save_image(recovery, image_name):
    recovery_image = torch.split(recovery, 1, dim=0)
    batch_num = len(recovery_image)

    if not os.path.exists('./results'):
        os.makedirs('./results')

    for ind in range(batch_num):
        utils.save_image(recovery_image[ind], './results/{}.png'.format(image_name[ind]))

def to_psnr(rect, gt):
    mse = F.mse_loss(rect, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    psnr_list = [-10.0 * math.log10(mse) for mse in mse_list]
    return psnr_list

def test(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()

    with torch.no_grad():
        for i, (images, gt_image, _) in enumerate(tqdm(test_loader)):

            images = [img_.cuda() for img_ in images]
            gt = gt_image.cuda()

            torch.cuda.synchronize()
            start_time = time.time()
            out = model(images)

            torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)

            myutils.eval_metrics(out, gt, psnrs, ssims)

    print("PSNR: %f, SSIM: %fn" %
          (psnrs.avg, ssims.avg))
    print("Time , " , sum(time_taken)/len(time_taken))

    return psnrs.avg


""" Entry Point """
def main(args):
    
    assert args.load_from is not None

    model_dict = model.state_dict()
    model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
