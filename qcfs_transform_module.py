import torch
import torch.nn as nn
# from models import *
from torch.autograd import Function

def isActivation(name):
    if 'relu' in name.lower() or 'qcfs' in name.lower():
        return True
    return False


def replace_activation_by_floor(model, t):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = QCFS(up=8., t=t)
    return model

# 对自动求导类实现，扩展算子（最基础的运算单元）
class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        #向下取整，转化为离散计算
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

qcfs = FloorLayer.apply

class QCFS(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        # 反向传播自动更新参数
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
    def forward(self, x):
        x = x / self.up
        x = qcfs(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x    