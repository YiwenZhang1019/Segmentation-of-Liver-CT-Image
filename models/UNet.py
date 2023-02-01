# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Res,unetUpR,unetConv2,unetUp
from init_weights import init_weights
from torchvision import models
import numpy as np

class UNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        #
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(filters[0], 1, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return F.sigmoid(d1)


class UNet1(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet1, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        #
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = Res(self.in_channels, filters[0],same_shape=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool1_3 = nn.MaxPool2d(kernel_size=4)
        self.maxpool1_4 = nn.MaxPool2d(kernel_size=8)

        self.conv2 = Res(filters[0], filters[1], same_shape=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2_4 = nn.MaxPool2d(kernel_size=4)

        self.conv3 = Res(filters[1], filters[2], same_shape=False)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = Res(filters[2], filters[3], same_shape=False)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = Res(filters[3], filters[4], same_shape=False)

        self.conv_cmb2=nn.Sequential(
            nn.Conv2d(filters[0]+filters[1],filters[1] , 3,1,1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True))

        self.conv_cmb3 = nn.Sequential(
            nn.Conv2d(filters[0]+filters[1]+filters[2], filters[2],3,1,1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True))

        self.conv_cmb4 = nn.Sequential(
            nn.Conv2d(filters[0]+filters[1]+filters[2]+filters[3], filters[3],3,1,1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True))

        # upsampling
        self.up_concat4 = unetUpR(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUpR(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUpR(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUpR(filters[1], filters[0], self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(filters[0], 1, 1, padding=0)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512
        maxpool1_3=self.maxpool1_3(conv1)
        maxpool1_4=self.maxpool1_4(conv1)

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256
        maxpool2_4=self.maxpool2_4(conv2)
        conv_com2=self.conv_cmb2(torch.cat([maxpool1, conv2], 1))

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128
        conv_com3=self.conv_cmb3(torch.cat([maxpool2, maxpool1_3,conv3], 1))

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64
        conv_com4=self.conv_cmb4(torch.cat([maxpool3, maxpool1_4,maxpool2_4,conv4], 1))

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv_com4)  # 128*64*128
        up3 = self.up_concat3(up4, conv_com3)  # 64*128*256
        up2 = self.up_concat2(up3, conv_com2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return F.sigmoid(d1)

class bishe(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(bishe, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        #
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = Res(self.in_channels, filters[0], same_shape=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = Res(filters[0], filters[1], same_shape=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = Res(filters[1], filters[2], same_shape=False)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = Res(filters[2], filters[3], same_shape=False)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = Res(filters[3], filters[4], same_shape=False)

        # upsampling
        self.up_concat4 = unetUpR(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUpR(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUpR(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUpR(filters[1], filters[0], self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(filters[0], 1, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return F.sigmoid(d1)
