from .unet_parts import *
from torch.autograd import Function
# from seed import seed_everything
from .modules import (
    ResidualConv,
    ASPP,
    ResUnetPlusPlu_AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


class UNet_ori(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_ori, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_FE(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet_FE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNet_DE(nn.Module):
    def __init__(self, n_classes=2 ,bilinear=True):
        super(UNet_DE, self).__init__()
        self.n_classes = n_classes
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# +
class UNet_DANN(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet_DANN, self).__init__()
        factor = 2 if bilinear else 2
        #=========================dann
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('h_fc1', nn.Linear(1024 // 8 * 64 * 64, 100))
#         self.domain_classifier.add_module('h_fc1', nn.Linear(512, 100))
        self.domain_classifier.add_module('h_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('h_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('h_fc2', nn.Linear(100, 3))
        self.domain_classifier.add_module('h_softmax', nn.Softmax(dim=1))
        
#         self.s_domain_classifier = nn.Sequential()
#         self.s_domain_classifier.add_module('s_fc1', nn.Linear(512, 100))
# #         self.s_domain_classifier.add_module('s_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
#         self.s_domain_classifier.add_module('s_bn1', nn.BatchNorm1d(100))
#         self.s_domain_classifier.add_module('s_relu1', nn.ReLU(True))
#         self.s_domain_classifier.add_module('s_fc2', nn.Linear(100, 2))
#         self.s_domain_classifier.add_module('s_softmax', nn.Softmax(dim=1))
        
#         self.l_domain_classifier = nn.Sequential()
#         self.l_domain_classifier.add_module('d_fc1', nn.Linear(512, 100))
# #         self.l_domain_classifier.add_module('d_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
#         self.l_domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
#         self.l_domain_classifier.add_module('d_relu1', nn.ReLU(True))
#         self.l_domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
#         self.l_domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(self, x5):
#         feature = x5.view(-1, 1024 // 2 * 64 * 64)
#         feature = nn.AdaptiveAvgPool2d((1,1))(x5)
        feature = torch.flatten(x5, 1)
#         print('x5.size()',x5.size())
#         print('feature.size()',feature.size())
#         reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(feature)
#         s_domain_output = self.s_domain_classifier(feature)
#         l_domain_output = self.l_domain_classifier(feature)
        return domain_output


# -

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.UNet_FE = UNet_FE(self.n_channels, self.n_classes, self.bilinear)
        self.UNet_DE = UNet_DE(self.n_classes)
        self.UNet_DANN = UNet_DANN()
        
        
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.UNet_FE(x)
        logits = self.UNet_DE(x1, x2, x3, x4, x5)
        domain_output = self.UNet_DANN(x5)
        return logits, domain_output, x5