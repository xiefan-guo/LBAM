import torch
import torch.nn as nn
import torchvision


# ---------------------
# VGG16 feature extract
# ---------------------
class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/data/guoxiefan/pytorch_ckpt/VGG/vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, images):
        results = [images]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def gram_matrix(feat):
    # ---------------------------------------------------------------------------------------
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    # ---------------------------------------------------------------------------------------
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)

    return gram


