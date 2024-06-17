import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)
        # self.border_output = nn.Conv2d(out_filters[1], 1, 1)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 2, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        
        
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        # self.border_output = nn.Conv2d(out_filters[0], 1, 1) # border
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.final2 = nn.Conv2d(out_filters[1], num_classes, 1)
        self.final3 = nn.Conv2d(out_filters[2], num_classes, 1)
        self.final4 = nn.Conv2d(out_filters[3], num_classes, 1)
        

        self.backbone = backbone

    def forward(self, inputs, need_fp=False):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        
        final = self.final(up1)
        # border_pred = self.border_output(up1)#(up2)
        
        
     
        # Final predictions at different scales
#         final4 = nn.functional.upsample(self.final4(up4), inputs.size()[2:], mode="bilinear")
#         final3 = nn.functional.upsample(self.final3(up3), inputs.size()[2:], mode="bilinear")
#         final2 = nn.functional.upsample(self.final2(up2), inputs.size()[2:], mode="bilinear")
#         final1 = nn.functional.upsample(self.final(up1), inputs.size()[2:], mode="bilinear")

        
#         if need_fp:
#             # 对feat4应用扰动，例如Dropout
#             # feat4_fp = nn.Dropout2d(0.5)(feat4)
#             noise = torch.randn_like(feat4) * 0.1
#             feat4_fp = feat4 + noise

#             # 使用扰动的feat4重新进行解码
#             up4_fp = self.up_concat4(feat4_fp, feat5)
#             up3_fp = self.up_concat3(feat3, up4_fp)
#             up2_fp = self.up_concat2(feat2, up3_fp)
#             up1_fp = self.up_concat1(feat1, up2_fp)

#             if self.up_conv != None:
#                 up1_fp = self.up_conv(up1_fp)

#             final_fp = self.final(up1_fp)

#             return final, final_fp#, border_pred
        
        
#         return final1, final2, final3, final4#, border_pred
        return final#,feat4

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
