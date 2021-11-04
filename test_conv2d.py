import torch
from opacus.grad_sample.grad_sample_module import GradSampleModule
import torch.nn as nn

x = torch.ones(1, 6, 1, 1)
x[0, 3:] = 2
layer = nn.Conv2d(6, 4, kernel_size=1, groups=2, bias=None)

m = GradSampleModule(layer)

y = m(x)
backprops = torch.ones_like(y)
y.backward(backprops)

