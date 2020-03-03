import os
import argparse

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image

from src.dataset import check_image_file
from src.generator import LBAM


parser = argparse.ArgumentParser()
parser.add_argument('--input_root', type=str, default='', help='input damaged image')
parser.add_argument('--mask_root', type=str, default='', help='input mask')
parser.add_argument('--output_root', type=str, default='results/test', help='output file name')
parser.add_argument('--pre_trained', type=str, default='', help='load pre-trained model')
parser.add_argument('--load_size', type=int, default=350, help='image loading size')
parser.add_argument('--crop_size', type=int, default=256, help='image training size')
args = parser.parse_args()

image_transforms = transforms.Compose([
    transforms.Resize(size=(args.crop_size, args.crop_size), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

mask_transforms = transforms.Compose([
    transforms.Resize(size=(args.crop_size, args.crop_size), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

if not check_image_file(args.input_root):
    print('Input file is not image file!')
elif not check_image_file(args.mask_root):
    print('Input mask is not image file!')
elif args.pre_trained == '':
    print('Please provide pre-trained model!')
else:
    image = image_transforms(Image.open(args.input_root).convert('RGB'))
    mask = mask_transforms(Image.open(args.mask_root).convert('RGB'))

    threshold = 0.5
    ones = mask >= threshold
    zeros = mask < threshold

    mask.masked_fill_(ones, 1.0)
    mask.masked_fill_(zeros, 0.0)

    mask = 1 - mask

    ground_truth = image

    sizes = image.size()
    image = image * mask
    image = torch.cat((image, mask[0].view(1, sizes[1], sizes[2])), dim=0)
    image = image.view(1, 4, sizes[1], sizes[2])

    mask = mask.view(1, sizes[0], sizes[1], sizes[2])

    generator = LBAM(4, 3)
    generator.load_state_dict(torch.load(args.pre_trained))
    for param in generator.parameters():
        param.requires_grad = False

    generator.eval()

    if torch.cuda.is_available():

        generator = generator.cuda()
        image = image.cuda()
        mask = mask.cuda()

    output = generator(image, mask)
    output_comp = output * (1 - mask) + image[:, 0:3, :, :] * mask

    save_image(output, '%s/test_output.png' % (args.output_root))


