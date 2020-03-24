import os
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from PIL import Image

from src.dataset import ImageDataset
from src.generator import LBAM
from src.evaluate import compute_psnr, compute_ssim


parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=4, help='workers for dataloader')
parser.add_argument('--pre_trained', type=str, default='', help='pre-trained models for fine-tuning')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--load_size', type=int, default=350, help='image loading size')
parser.add_argument('--crop_size', type=int, default=256, help='image training size')
parser.add_argument('--image_root', type=str, default='')
parser.add_argument('--mask_root', type=str, default='')
parser.add_argument('--result_root', type=str, default='./results/CelebA', help='train result')
parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
parser.add_argument('--number_eval', type=int, default=60, help='number of batches eval')
args = parser.parse_args()


is_cuda = torch.cuda.is_available()
if is_cuda:
    print('Cuda is available!')
    cudnn.benchmark = True

if not os.path.exists(args.result_root):
    os.makedirs(args.result_root)

image_dataset = ImageDataset(
    args.image_root, args.mask_root, (args.load_size, args.load_size), (args.crop_size, args.crop_size)
)
data_loader = DataLoader(
    image_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=False
)

generator = LBAM(4, 3)

if args.pre_trained != '':
    generator.load_state_dict(torch.load(args.pre_trained)['generator'])
else:
    print('Please provide pre-trained model!')

for param in generator.parameters():
    param.requires_grad = False

if is_cuda:
    generator = generator.cuda()


sum_psnr = 0.0
sum_ssim = 0.0
l1_loss = 0.0
sum_time = 0.0
count = 0
start_time = time.time()

for epoch in range(1, args.train_epochs):

    if count >= args.number_eval:
        break

    generator.eval()
    for _, (input_images, ground_truths, masks) in enumerate(data_loader):

        if count >= args.number_eval:
            break

        count = count + 1
        if is_cuda:
            input_images = input_images.cuda()
            ground_truths = ground_truths.cuda()
            masks = masks.cuda()

        outputs = generator(input_images, masks)

        outputs = outputs.data.cpu()
        ground_truths = ground_truths.data.cpu()
        masks = masks.data.cpu()

        damaged_images = ground_truths * masks + (1 - masks)
        outputs_comps = ground_truths * masks + outputs * (1 - masks)

        psnr = compute_psnr(ground_truths, outputs_comps)
        sum_psnr += psnr
        print(count, ' psnr: ', psnr)

        ssim = compute_ssim(ground_truths * 255, outputs_comps * 255).item()
        sum_ssim += ssim
        print(count, ' ssim: ', ssim)

        l1 = nn.L1Loss()(ground_truths, outputs_comps).item()
        l1_loss += l1
        print(count, ' l1_loss: ', l1)

        sizes = ground_truths.size()
        save_images = torch.Tensor(sizes[0] * 4, sizes[1], sizes[2], sizes[3])

        for i in range(sizes[0]):
            save_images[4 * i] = 1 - masks[i]
            save_images[4 * i + 1] = damaged_images[i]
            save_images[4 * i + 2] = outputs[i]
            save_images[4 * i + 3] = ground_truths[i]

        save_image(save_images, os.path.join(args.result_root, 'result_{}.png'.format(count)), nrow=args.batch_size)

        # -----------------------------------------------------------------
        # make sub_dirs to save mask GT results and input and damaged images
        # -----------------------------------------------------------------
        os.makedirs(('%s/mask' % args.result_root), exist_ok=True)
        os.makedirs(('%s/damaged' % args.result_root), exist_ok=True)
        os.makedirs(('%s/result' % args.result_root), exist_ok=True)
        os.makedirs(('%s/ground-truth' % args.result_root), exist_ok=True)

        for i in range(sizes[0]):
        
            save_image(save_images[4 * i], args.result_root + '/mask/mask{}_{}.png'.format(count, i))
            save_image(save_images[4 * i + 1], args.result_root + '/damaged/damaged{}_{}.png'.format(count, i))
            save_image(save_images[4 * i + 2], args.result_root + '/result/result{}_{}.png'.format(count, i))
            save_image(save_images[4 * i + 3], args.result_root + '/ground-truth/ground-truth{}_{}.png'.format(count, i))


end_time = time.time()
sum_time = end_time - start_time

print('count: ', count)
print('avgrage l1 loss: ', l1_loss / count)
print('average psnr: ', sum_psnr / count)
print('average ssim: ', sum_ssim / count)
print('average time cost: ', sum_time / count)



