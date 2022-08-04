# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/7 19:36
import torch
from torch import nn
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, X):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(X)

def Spectral_Normalization(m):
    for name, layer in m.named_children():
        m.add_module(name, Spectral_Normalization(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m

