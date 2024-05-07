import torch
import torch.nn as nn
from torchvision import models
from loss.CL1 import L1_Charbonnier_loss
from loss.SSIMLoss import SSIMLoss
from loss.Perceptual import *
from ssim_msssim import MS_SSIM
class VGG_loss(nn.Module):
    def __init__(self, model):
        super(VGG_loss, self).__init__()
        self.features = nn.Sequential(*list(model.children())[0][:-3])
    def forward(self, x):
        return self.features(x)

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
class combinedloss(nn.Module):
    def __init__(self):
        super(combinedloss, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG model is loaded")
        self.vggloss = VGG_loss(vgg)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.mssim = MS_SSIM()

    def forward(self, out, label):

        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.mseloss(out, label)
        vgg_loss = self.l1loss(inp_vgg, label_vgg)
        ssim_loss = self.ssim_loss(out,label)
        # ssim_loss = self.mssim(out,label)

        total_loss = mse_loss + vgg_loss+ssim_loss

        return total_loss
