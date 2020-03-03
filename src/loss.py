import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from src.utils import VGG16FeatureExtractor
from src.utils import gram_matrix


def calc_gradient_penalty(discriminator, real_samples, fake_samples, masks, cuda, lambda_gp):

    batch_size = real_samples.size(0)
    alpha = torch.randn(batch_size, 1, 1, 1)
    if cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    if cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    dis_interpolates = discriminator(interpolates, masks)

    grad_outputs = torch.ones(dis_interpolates.size())
    if cuda:
        grad_outputs = grad_outputs.cuda()

    gradients = torch.autograd.grad(
        outputs=dis_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

    return gradient_penalty


def total_variation_loss(image):
    # ---------------------------------------------------------------
    # shift one pixel and get difference (for both x and y direction)
    # ---------------------------------------------------------------
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))

    return loss


def discriminator_loss(log_dir, count, d_real, d_fake, gp):

    writer = SummaryWriter(log_dir)

    loss = -1.0 * d_real + d_fake + gp
    writer.add_scalar('D loss', loss.item(), count)
    writer.close()

    return loss


def generator_loss(log_dir, inputs, masks, outputs, ground_truths, count, extractor, loss_adversarial):

    l1 = nn.L1Loss()
    writer = SummaryWriter(log_dir)
    # extractor = VGG16FeatureExtractor()
    # if is_cuda:
    #     extractor = extractor.cuda()

    comp = masks * inputs + (1 - masks) * outputs

    loss_hole = l1((1 - masks) * outputs, (1 - masks) * ground_truths)
    loss_valid = l1(masks * outputs, masks * ground_truths)

    if outputs.size(1) == 3:
        feat_comp = extractor(comp)
        feat_output = extractor(outputs)
        feat_gt = extractor(ground_truths)
    elif outputs.size(1) == 1:
        feat_comp = extractor(torch.cat([comp] * 3, dim=1))
        feat_output = extractor(torch.cat([outputs] * 3, dim=1))
        feat_gt = extractor(torch.cat([ground_truths] * 3, dim=1))
    else:
        raise ValueError('only gray rgb')

    loss_perceptual = 0.0
    for i in range(3):
        loss_perceptual += l1(feat_output[i], feat_gt[i])
        loss_perceptual += l1(feat_comp[i], feat_gt[i])

    loss_style = 0.0
    for i in range(3):
        loss_style += l1(gram_matrix(feat_output[i]),
                              gram_matrix(feat_gt[i]))
        loss_style += l1(gram_matrix(feat_comp[i]),
                              gram_matrix(feat_gt[i]))

    total_loss = loss_hole * 6.0 + loss_valid + loss_perceptual * 0.05 + loss_style * 120 - 0.1 * loss_adversarial

    writer.add_scalar('G hole loss', loss_hole.item(), count)
    writer.add_scalar('G valid loss', loss_valid.item(), count)
    writer.add_scalar('G perceptual loss', loss_perceptual.item(), count)
    writer.add_scalar('G style loss', loss_style.item(), count)
    writer.add_scalar('G adversarial loss', loss_adversarial.item(), count)
    writer.add_scalar('G total loss', total_loss.item(), count)

    writer.close()

    return total_loss