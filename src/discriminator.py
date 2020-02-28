import torch
import torch.nn as nn


# -------------
# discriminator
# -------------
class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, bn=True):

            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.discriminator_valid = nn.Sequential(
            *discriminator_block(in_channels, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512),
            *discriminator_block(512, 512)
        )

        self.discriminator_masked = nn.Sequential(
            *discriminator_block(in_channels, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512),
            *discriminator_block(512, 512)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4),
            # nn.Sigmoid()  # author's version
        )

    def forward(self, images, masks):

        valid = self.discriminator_valid(images * masks)
        masked = self.discriminator_masked(images * (1 - masks))

        total =torch.cat((valid, masked), dim=1)

        return self.fusion(total).view(images.size(0), -1)
