import torch
from torch import nn
import numpy as np
import scipy

import torch.nn.functional as F

class Blurkernel(nn.Module):
    def __init__(self, kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )
        self.pad = nn.ReflectionPad2d(self.kernel_size//2)

        self.weights_init()

    def forward(self, x):
        return self.seq(x)
    
    def transpose(self, y):
        conv = self.seq[1]
        w = conv.weight
        wT = torch.flip(w, dims=[-1, -2])
        y = self.pad(y)
        z = F.conv2d(y, wT, stride=1, padding=0, bias=None, groups=3)

        return z

    def weights_init(self):
        n = np.zeros((self.kernel_size, self.kernel_size))
        n[self.kernel_size // 2,self.kernel_size // 2] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
        k = torch.from_numpy(k)
        self.k = k
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k