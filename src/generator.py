import math

import torch
import torch.nn as nn
import torchvision

from torch.nn.parameter import Parameter


# --------------
# weight initial
# --------------
def weights_init(init_type='gaussian'):

    def init_func(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):

            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


# --------------------------------------------------
# asymmetric gaussian-shaped activation function g_A
# --------------------------------------------------
class GaussianActivation(nn.Module):

    def __init__(self, a, mu, gamma_l, gamma_r):
        super(GaussianActivation, self).__init__()

        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.gamma_l = Parameter(torch.tensor(gamma_l, dtype=torch.float32))
        self.gamma_r = Parameter(torch.tensor(gamma_r, dtype=torch.float32))

    def forward(self, input_features):

        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.gamma_l.data = torch.clamp(self.gamma_l.data, 0.5, 2.0)
        self.gamma_r.data = torch.clamp(self.gamma_r.data, 0.5, 2.0)

        left = input_features < self.mu
        right = input_features >= self.mu

        g_A_left = self.a * torch.exp(-self.gamma_l * (input_features - self.mu) ** 2)
        g_A_left.masked_fill_(right, 0.0)

        g_A_right = 1 + (self.a - 1) * torch.exp(-self.gamma_r * (input_features - self.mu) ** 2)
        g_A_right.masked_fill_(left, 0.0)

        g_A = g_A_left + g_A_right

        return g_A


# ----------------------
# mask updating function
# ----------------------
class MaskUpdate(nn.Module):

    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()

        self.func = nn.ReLU(True)
        self.alpha = alpha

    def forward(self, input_masks):

        return torch.pow(self.func(input_masks), self.alpha)


# --------------------------------------
# learnable forward attention conv layer
# --------------------------------------
class ForwardAttentionLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(ForwardAttentionLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        if in_channels == 4:
            self.mask_conv = nn.Conv2d(3, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        else:
            self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.conv.apply(weights_init())
        self.mask_conv.apply(weights_init())

        self.gaussian_activation = GaussianActivation(a=1.1, mu=2.0, gamma_l=1.0, gamma_r=1.0)
        self.mask_update = MaskUpdate(alpha=0.8)

    def forward(self, input_features, input_masks):

        conv_features = self.conv(input_features)
        conv_masks = self.mask_conv(input_masks)

        gaussian_masks = self.gaussian_activation(conv_masks)
        output_features = conv_features * gaussian_masks

        output_masks = self.mask_update(conv_masks)

        return output_features, output_masks, conv_features, gaussian_masks


# -----------------
# forward attention
# -----------------
class ForwardAttention(nn.Module):

    def __init__(self, in_channels, out_channels, bn=False, sample='down-4', activ='leaky', bias=False):
        super(ForwardAttention, self).__init__()

        if sample == 'down-4':
            self.conv = ForwardAttentionLayer(in_channels, out_channels, 4, 2, 1, bias=bias)
        elif sample == 'down-5':
            self.conv = ForwardAttentionLayer(in_channels, out_channels, 5, 2, 2, bias=bias)
        elif sample == 'down-7':
            self.conv = ForwardAttentionLayer(in_channels, out_channels, 7, 2, 3, bias=bias)
        elif sample == 'down-3':
            self.conv = ForwardAttentionLayer(in_channels, out_channels, 3, 2, 1, bias=bias)
        else:
            self.conv = ForwardAttentionLayer(in_channels, out_channels, 3, 1, 1, bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, False)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass

    def forward(self, input_features, input_masks):

        output_features, output_masks, conv_features, gaussian_masks = self.conv(input_features, input_masks)

        if hasattr(self, 'bn'):
            output_features = self.bn(output_features)
        if hasattr(self, 'activ'):
            output_features = self.activ(output_features)

        return output_features, output_masks, conv_features, gaussian_masks


# --------------------------------------
# learnable reverse attention conv layer
# --------------------------------------
class ReverseAttentionLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=False):
        super(ReverseAttentionLayer, self).__init__()

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.mask_conv.apply(weights_init())

        self.gaussian_activation = GaussianActivation(a=1.1, mu=2.0, gamma_l=1.0, gamma_r=1.0)
        self.mask_update = MaskUpdate(alpha=0.8)

    def forward(self, input_masks):

        conv_masks = self.mask_conv(input_masks)
        gaussian_masks = self.gaussian_activation(conv_masks)

        output_masks = self.mask_update(conv_masks)

        return output_masks, gaussian_masks


# -----------------
# reverse attention
# -----------------
class ReverseAttention(nn.Module):

    def __init__(self, in_channels, out_channels, bn=False, activ='leaky', kernel_size=4, stride=2, padding=1,
                 output_padding=0, dilation=1, groups=1, bias=False, bn_channels=512):
        super(ReverseAttention, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

        self.conv.apply(weights_init())

        if bn:
            self.bn = nn.BatchNorm2d(bn_channels)

        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, False)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass

    def forward(self, ec_features, dc_features, input_masks_attention):

        conv_dc_features = self.conv(dc_features)

        output_features = torch.cat((ec_features, conv_dc_features), dim=1)
        output_features = output_features * input_masks_attention

        if hasattr(self, 'bn'):
            output_features = self.bn(output_features)
        if hasattr(self, 'activ'):
            output_features = self.activ(output_features)

        return output_features


# --------------------------------------
# Learnable Bidirectional Attention Maps
# --------------------------------------
class LBAM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LBAM, self).__init__()

        # -------------------------------------------
        # default: kernel_size=4, stride=2, padding=1
        # bias=False in default ReverseAttention
        # -------------------------------------------
        self.ec_1 = ForwardAttention(in_channels, 64, bn=False)
        self.ec_2 = ForwardAttention(64, 128)
        self.ec_3 = ForwardAttention(128, 256)
        self.ec_4 = ForwardAttention(256, 512)

        for _ in range(5, 8):
            name = 'ec_{:d}'.format(_)
            setattr(self, name, ForwardAttention(512, 512))

        self.reverse_attention_layer_1 = ReverseAttentionLayer(3, 64)
        self.reverse_attention_layer_2 = ReverseAttentionLayer(64, 128)
        self.reverse_attention_layer_3 = ReverseAttentionLayer(128, 256)
        self.reverse_attention_layer_4 = ReverseAttentionLayer(256, 512)
        self.reverse_attention_layer_5 = ReverseAttentionLayer(512, 512)
        self.reverse_attention_layer_6 = ReverseAttentionLayer(512, 512)

        self.dc_1 = ReverseAttention(512, 512, bn_channels=1024)
        self.dc_2 = ReverseAttention(512 * 2, 512, bn_channels=1024)
        self.dc_3 = ReverseAttention(512 * 2, 512, bn_channels=1024)
        self.dc_4 = ReverseAttention(512 * 2, 256, bn_channels=512)
        self.dc_5 = ReverseAttention(256 * 2, 128, bn_channels=256)
        self.dc_6 = ReverseAttention(128 * 2, 64, bn_channels=128)
        self.dc_7 = nn.ConvTranspose2d(64 * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, input_images, input_masks):

        ec_features_1, ec_masks_1, skip_features_1, ec_gaussian_1 = self.ec_1(input_images, input_masks)
        ec_features_2, ec_masks_2, skip_features_2, ec_gaussian_2 = self.ec_2(ec_features_1, ec_masks_1)
        ec_features_3, ec_masks_3, skip_features_3, ec_gaussian_3 = self.ec_3(ec_features_2, ec_masks_2)
        ec_features_4, ec_masks_4, skip_features_4, ec_gaussian_4 = self.ec_4(ec_features_3, ec_masks_3)
        ec_features_5, ec_masks_5, skip_features_5, ec_gaussian_5 = self.ec_5(ec_features_4, ec_masks_4)
        ec_features_6, ec_masks_6, skip_features_6, ec_gaussian_6 = self.ec_6(ec_features_5, ec_masks_5)
        ec_features_7, ec_masks_7, skip_features_7, ec_gaussian_7 = self.ec_7(ec_features_6, ec_masks_6)

        dc_masks_1, dc_gaussian_1 = self.reverse_attention_layer_1(1 - input_masks)
        dc_masks_2, dc_gaussian_2 = self.reverse_attention_layer_2(dc_masks_1)
        dc_masks_3, dc_gaussian_3 = self.reverse_attention_layer_3(dc_masks_2)
        dc_masks_4, dc_gaussian_4 = self.reverse_attention_layer_4(dc_masks_3)
        dc_masks_5, dc_gaussian_5 = self.reverse_attention_layer_5(dc_masks_4)
        dc_masks_6, dc_gaussian_6 = self.reverse_attention_layer_6(dc_masks_5)

        concat_gaussian_6 = torch.cat((ec_gaussian_6, dc_gaussian_6), dim=1)
        dc_features_1 = self.dc_1(skip_features_6, ec_features_7, concat_gaussian_6)

        concat_gaussian_5 = torch.cat((ec_gaussian_5, dc_gaussian_5), dim=1)
        dc_features_2 = self.dc_2(skip_features_5, dc_features_1, concat_gaussian_5)

        concat_gaussian_4 = torch.cat((ec_gaussian_4, dc_gaussian_4), dim=1)
        dc_features_3 = self.dc_3(skip_features_4, dc_features_2, concat_gaussian_4)

        concat_gaussian_3 = torch.cat((ec_gaussian_3, dc_gaussian_3), dim=1)
        dc_features_4 = self.dc_4(skip_features_3, dc_features_3, concat_gaussian_3)

        concat_gaussian_2 = torch.cat((ec_gaussian_2, dc_gaussian_2), dim=1)
        dc_features_5 = self.dc_5(skip_features_2, dc_features_4, concat_gaussian_2)

        concat_gaussian_1 = torch.cat((ec_gaussian_1, dc_gaussian_1), dim=1)
        dc_features_6 = self.dc_6(skip_features_1, dc_features_5, concat_gaussian_1)

        dc_features_7 = self.dc_7(dc_features_6)

        output = (self.tanh(dc_features_7) + 1) / 2

        return output