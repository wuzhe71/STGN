import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import pdb

def normailize(in_):
    in_ = F.relu(in_)
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-16)
    
def NSS(density_pred, gt):
    density_pred = normailize(density_pred)
    mean, std = torch.std_mean(density_pred, dim=(2, 3), keepdim=True)
    density_pred = (density_pred - mean) / (std + 1e-18)
    score = density_pred*gt 
    return score.mean(dim=(2, 3)).sum()


def SSIM(density_pred, density):
    density_pred = normailize(density_pred)
    density = normailize(density)
    _, _, h, w = density.shape
    x, sigma_x = torch.var_mean(density_pred, dim=(2, 3), keepdim=True)
    y, sigma_y = torch.var_mean(density, dim=(2, 3), keepdim=True)
    sigma_xy = ((density_pred-x)*(density-y)).sum(dim=(2, 3), keepdim=True) / (h*w-1)
    alpha = 4 * x * y * sigma_xy
    beta = (x*x + y*y)*(sigma_x + sigma_y)
    score = alpha / (beta + 1e-16)
    return score.sum()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss/gauss.sum()
    return gauss


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=(2, 3)).sum()


# class SSIM(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)

#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)

#             self.window = window
#             self.channel = channel

#         return _ssim(img1, img2, window, self.window_size, channel)


def ssim(img1, img2, window_size=15):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)


class PL(nn.Module):
    # pyramid loss
    def __init__(self):
        super(PL, self).__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=5)
        self.pool3 = nn.AvgPool2d(kernel_size=7)

    def forward(self, pred, label):
        assert pred.size() == label.size()
        n, t, c, h, w = pred.shape
        pred = pred.reshape((-1, c, h, w))
        label = label.reshape((-1, c, h, w))
        N = n*t
        loss0 = torch.sum(torch.abs(pred - label)) / N
        
        pred1 = self.pool1(pred)*9
        label1 = self.pool1(label)*9
        loss1 = torch.sum(torch.abs(pred1 - label1)) / N

        pred2 = self.pool2(pred)*25
        label2 = self.pool2(label)*25
        loss2 = torch.sum(torch.abs(pred2 - label2)) / N

        pred3 = self.pool3(pred)*49
        label3 = self.pool3(label)*49
        loss3 = torch.sum(torch.abs(pred3 - label3)) / N
        

        return (loss0 + loss1 + loss2 + loss3)/4