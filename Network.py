# object localization and edge refinement network

import torch
import torch.nn.functional as F
from torch import nn
from pvtv2 import pvt_v2_b2

# Information Multiple Selection Module
class IMSModule(nn.Module):
    def __init__(self, channel):
        super(IMSModule, self).__init__()
        self.im = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1), nn.Sigmoid())
        self.channel = channel

        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                  nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                  nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                  nn.PReLU())

    def forward(self, x):
        im = self.im(x)

        t1 = torch.zeros_like(im)
        t2 = torch.zeros_like(im)
        t3 = torch.zeros_like(im)

        k1 = torch.ones_like(im)
        k2 = torch.ones_like(im)
        k3 = torch.ones_like(im)

        sel1 = torch.where(im >= 0.3, k1, t1)
        sel2 = torch.where(im >= 0.5, k2, t2)
        sel3 = torch.where(im >= 0.7, k3, t3)

        conv1 = self.conv1(x * sel1)
        conv2 = self.conv2(x * sel2)
        conv3 = self.conv3(x * sel3)

        return conv1 * conv2 * conv3


# Edge generation module
class EGM(nn.Module):
    def __init__(self, channel):
        super(EGM, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(4, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                   nn.PReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                  nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                  nn.PReLU())
        self.conv3_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                  nn.PReLU(),nn.Conv2d(channel, 1, kernel_size=1))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                     nn.PReLU(), nn.Conv2d(channel, 1, kernel_size=1))

    def forward(self, x, sal):

        conv0 = self.conv0(torch.cat((x, sal), 1))
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3_1 = self.conv3_1(conv2)
        conv3_2 = self.conv3_2(conv2)

        return conv3_1, conv3_2


class ERM(nn.Module):
    def __init__(self, channel):
        super(ERM, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(6, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                   nn.PReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                   nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                   nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(32, channel),
                                     nn.PReLU(), nn.Conv2d(channel, 1, kernel_size=1))

    def forward(self, x, sal, e1, e2):
        conv0 = self.conv0(torch.cat((x, sal, e1, e2), 1))
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        return conv3


class OLER(nn.Module):
    def __init__(self):
        super(OLER, self).__init__()
        self.backbone = pvt_v2_b2()
        path = './PyTorch Pretrained/PVT/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.down4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU())
        self.down3 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU())
        self.down2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU())
        self.down1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU())

        self.fuse1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU())
        self.sal1 = nn.Conv2d(64, 1, kernel_size=1)

        self.IMS1 = IMSModule(64)
        self.IMS2 = IMSModule(64)
        self.IMS3 = IMSModule(64)
        self.IMS4 = IMSModule(64)

        self.EGM = EGM(64)
        self.ERM = ERM(64)

    def forward(self, x):
        pvt = self.backbone(x)
        layer1 = pvt[0]
        layer2 = pvt[1]
        layer3 = pvt[2]
        layer4 = pvt[3]

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)

        ndown4 = self.IMS4(down4)
        ndown3 = self.IMS3(down3)
        ndown2 = self.IMS2(down2)
        ndown1 = self.IMS1(down1)

        ndown4 = F.interpolate(ndown4, size=ndown1.size()[2:], mode='bilinear', align_corners=True)
        ndown3 = F.interpolate(ndown3, size=ndown1.size()[2:], mode='bilinear', align_corners=True)
        ndown2 = F.interpolate(ndown2, size=ndown1.size()[2:], mode='bilinear', align_corners=True)
        fuse1 = self.fuse1(torch.cat((ndown4, ndown3, ndown2, ndown1), 1))
        sal1 = self.sal1(fuse1)

        sal1 = F.interpolate(sal1, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge1, edge2 = self.EGM(x, torch.sigmoid(sal1))
        fsal = self.ERM(x, torch.sigmoid(sal1), torch.sigmoid(edge1), torch.sigmoid(edge2))

        return sal1, fsal, edge1, edge2