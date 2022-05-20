import numpy as np
import torch
import pdb
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


def sort_seqs_by_len(seqs, seq_len):
    sort_idx = torch.argsort(seq_len, dim=0, descending=True)
    seq_len_sort = seq_len[sort_idx]
    seqs_sort = []
    for seq in seqs:
        seqs_sort.append(seq[sort_idx])

    return seqs_sort, seq_len_sort


def gauss2d(shape, center, gamma, out_shape=None):
    H, W = shape
    if out_shape is None:
        Ho = H
        Wo = W
    else:
        Ho, Wo = out_shape
    x, y = np.array(range(Wo)), np.array(range(Ho))
    x, y = np.meshgrid(x, y)
    x, y = x.astype(float)/Wo, y.astype(float)/Ho
    x0, y0 = float(center[0])/W, float(center[1])/H
    # Gaussian kernel centered in (x0, y0)
    G = np.exp(-(1/2)*(((x - x0)*gamma[0])**2 + ((y - y0)*gamma[1])**2))
    return G/np.sum(G)  # normalized so it sums to 1


def density_map(gt, gammas, mask=None):
    D = np.zeros(gt.shape)
    centers = np.nonzero(gt)
    for i in range(int(gt.sum())):
        y = centers[0][i]
        x = centers[1][i]
        dense = gauss2d(gt.shape, (x, y), gammas[i], out_shape=gt.shape)
        if mask is not None:
            dense *= mask
        dense /= dense.sum()
        D += dense
    return D


def gaussian_filter_density(gt, gamma=3, k=4, adaptive=False, mask=None):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=k)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if adaptive:
            if gt_count > 1:
                sigma = (np.array([distances[i][j] for j in range(1, k)])) * 0.1
            else:
                sigma = np.average(np.array(gt.shape))/2./2
        else:
            sigma = gamma
        map = gaussian_filter(pt2d, sigma, mode='constant')
        if mask is not None:
            map *= mask
        map = map / map.sum()
        density += map
    return density


class Gaussian(nn.Module):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels
        # gaussian kernel
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = torch.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= gauss.sum()
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(in_channels, 1, kernel_size, kernel_size).contiguous())
            windows.append(window)
        kernels = torch.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)
        
        self.gkernel = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.gkernel.weight = torch.nn.Parameter(weight)

        if froze: self.frozePara()

    def forward(self, dotmaps):
        gaussianmaps = self.gkernel(dotmaps)
        return gaussianmaps
    
    def frozePara(self):
        for para in self.parameters():
            para.requires_grad = False


class SumPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(SumPool2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        if type(kernel_size) is not int:
            self.area = kernel_size[0] * kernel_size[1]
        else:
            self.area = kernel_size * self.kernel_size
    
    def forward(self, dotmap):
        return self.avgpool(dotmap) * self.area


class Gaussianlayer(nn.Module):
    def __init__(self, sigma=None, kernel_size=11):
        super(Gaussianlayer, self).__init__()
        if sigma == None:
            sigma = [4]
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size//2, froze=True)
    
    def forward(self, dotmaps):
        denmaps = self.gaussian(dotmaps)
        return denmaps
    