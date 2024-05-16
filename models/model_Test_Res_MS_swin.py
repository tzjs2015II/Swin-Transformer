import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_,PatchEmbed, Mlp, to_ntuple

from typing import Optional

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
thresh = 0.01  # 0.5 # neuronal threshold 原始要有差距
thresh_ql = 0.9 # 快捷链接参数

lens = 0.5  # 0.5 # hyper-parameters of approximate function
decay = 0.25  # 0.25 # decay constants
num_classes = 10
time_window = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from einops import rearrange

# define approximate firing function

# SNN 激活函数 
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): # 在向前传播的时候剪裁掉小于阈值的浮点值。
        # 保留此时的张量用于后向传播
        ctx.save_for_backward(input)
        # 返回输入中大于thresh[阈值]的浮点值
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output): # todo 一些计算细节
        # 前向传播过程中被保留的张量
        (input,) = ctx.saved_tensors
        # print(str(input))
        # 保存上一层梯度
        grad_input = grad_output.clone()
        # 判断输入值减去阈值后是否小于[模拟函数超参数]
        temp = abs(input - thresh) < lens
        # 布尔值转化为浮点值，然后进行缩放
        temp = temp / (2 * lens)
        
        # 输入值 * 转换后的布尔值，显然没达到阈值的值会被直接归0，也对应没有达到阈值的脉冲值不会对权重产生影响
        return grad_input * temp.float()

# SNN 激活函数 
class ActFun_ql(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input): # 在向前传播的时候剪裁掉小于阈值的浮点值。
        # 保留此时的张量用于后向传播
        ctx.save_for_backward(input)
        # 返回输入中大于thresh[阈值]的浮点值
        return input.gt(thresh_ql).float()

    @staticmethod
    def backward(ctx, grad_output): 
        # 前向传播过程中被保留的张量
        (input,) = ctx.saved_tensors
        # 保存上一层梯度
        grad_input = grad_output.clone()
        # 判断输入值减去阈值后是否小于[模拟函数超参数]
        temp = abs(input - thresh_ql) < lens
        # 布尔值转化为浮点值，然后进行缩放
        temp = temp / (2 * lens)
        
        # 输入值 * 转换后的布尔值，显然没达到阈值的值会被直接归0，也对应没有达到阈值的脉冲值不会对权重产生影响
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update

act_fun_ql = ActFun_ql.apply

# 4 被基础层调用 膜电位更新
class mem_update(nn.Module):
    # LIF Layer
    def __init__(self):
        super(mem_update, self).__init__()

    # todo 不同的添加thresh的方式 需要不同的设定方式
    def forward(self, x):
        # 初始化和输入特征X同大小的全0张量
        mem = torch.zeros_like(x[0]).to(device) # 此时的处理又把时间切掉了，mem.shape = torch.Size([68, 64, 16, 16])
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        # 初始为0
        mem_old = 0
        # timestep此时固定为1，可能是因为这本身是一个静态图像数据集
        for i in range(time_window):
            # 此时time_window固定为1，意味着只会执行一次膜更新
            if i >= 1:
                # 当前膜电位 = 之前膜电位 * 衰减系数 * （1-激活值【副本】） + 特征[i]  实质是LIF模型
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
            else:
                #todo 首次执行此处，让men获取x初始的时间切片
                mem = x[i]
            # 激活值的被激活函数处理后的膜电位
            # 
            spike = act_fun(mem)
            # 保留之前的膜电位，此时膜电位的更新是以timestep的次数作为标准，因为只循环一次，所以此时mem_old恒定为0
            mem_old = mem.clone()
            # 多少timestep，output的数组长度就有多长
            output[i] = spike
        return output

# 4 快捷链接膜电位
class mem_update_ql(nn.Module):
    # LIF Layer
    def __init__(self):
        super(mem_update_ql, self).__init__()

    # todo 不同的添加thresh的方式 需要不同的设定方式
    def forward(self, x):
        # 初始化和输入特征X同大小的全0张量
        mem = torch.zeros_like(x[0]).to(device) # 此时的处理又把时间切掉了，mem.shape = torch.Size([68, 64, 16, 16])
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        # 初始为0
        mem_old = 0
        # timestep此时固定为1，可能是因为这本身是一个静态图像数据集
        for i in range(time_window):
            # 此时time_window固定为1，意味着只会执行一次膜更新
            if i >= 1:
                # 当前膜电位 = 之前膜电位 * 衰减系数 * （1-激活值【副本】） + 特征[i]  实质是LIF模型
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
            else:
                #todo 首次执行此处，让men获取x初始的时间切片
                mem = x[i]
            # 激活值的被激活函数处理后的膜电位
            # 
            spike = act_fun_ql(mem)
            # 保留之前的膜电位，此时膜电位的更新是以timestep的次数作为标准，因为只循环一次，所以此时mem_old恒定为0
            mem_old = mem.clone()
            # 多少timestep，output的数组长度就有多长
            output[i] = spike
        return output

class Snn_Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,# 当设置 dilation 参数时，卷积核中的元素之间会存在一定的间隔，即在计算时跳过一些位置。这可以看作是在卷积核中插入一些零元素，从而调整了卷积核的感受野。
        groups=1,
        bias=True,
        padding_mode="zeros",
        marker="b",
    ):
        super(Snn_Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (input.size()[3] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1 #通过填充步长与卷积核计算特征图高度与宽度
        w = (input.size()[4] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        
        # 创建一个timestep的时间序列，一个五维的全零张量
        c1 = torch.zeros(
            time_window,
            input.size()[1], 
            self.out_channels, 
            h, 
            w, 
            device=input.device
        )
        # 处理一个timestep的时间序列，每一个timestep都进行一次2D卷积,也许可以直接用一个3D卷积代替此处的2D循环卷积
        for i in range(time_window):
            c1[i] = F.conv2d(
                input[i],
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return c1

# 5 紧随着被基础层调用，在卷积后
class batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(
            # 
            num_features
        )  # input (N,C,D,H,W) 进行C-dimension batch norm on (N,D,H,W) slice. spatio-temporal Batch Normalization

    def forward(self, input):
        y = (
            input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        )  # 可以使用permute实现？ # y = input.permute(1,2,0).contiguous
        # y = self.bn(y,4)
        y = self.bn(y)
        return (
            y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)
        )  # 原始输入是(T,N,C,H,W) BN处理时转变为(N,C,T,H,W)
        
# 5_1 被batch_norm_2d调用 
class BatchNorm3d1(torch.nn.BatchNorm3d):
    def reset_parameters(self):
            # 重置运行时统计信息，包括运行时均值和方差等
        self.reset_running_stats()
        # BN层是否是可学习的，默认为true，执行
        if self.affine:
            # 膜电位 初始化可学习参数 weight
            nn.init.constant_(self.weight, thresh)
            # 偏置项清零
            nn.init.zeros_(self.bias)

# 6 紧随着被基础层调用，在卷积后
class batch_norm_2d1(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        # 将维度0和2进行交换，然后交换后的结果再进行维度0和1的交换。这样就实现了将维度2移到了最前面，而维度0和1的位置互换。
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        # y = self.bn(y,4)
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)

# 6_1 被batch_norm_2d1调用 
class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            # 初始化的时候，膜电位乘一个0.2
            nn.init.constant_(self.weight, 0.2 * thresh)
            nn.init.zeros_(self.bias)


class CSA(nn.Module):
    def __init__(
        self, 
        timeWindows, 
        channels, 
        stride=1, 
        fbs=False, 
        #  通道注意力模块中的压缩比例（compression ratio）。用于控制通过通道注意力机制后输出通道的数量。
        c_ratio=16, 
        t_ratio=1
    ):
        super(CSA, self).__init__()

        # CSA层有激活函数
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        # self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        out = self.ca(x) * x  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out

# todo 通道注意力(对激活值的权重)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=5):
        super(ChannelAttention, self).__init__()
        # 每个通道平均压缩为1，即为一个像素点 
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, "b f c h w -> b c f h w")
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        out = rearrange(out, "b c f h w -> b f c h w")
        return out

# todo 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = x.shape[2]
        x = rearrange(x, "b f c h w -> b (f c) h w")
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # 链接张量
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = x.unsqueeze(1)

        return self.sigmoid(x)





def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask: Optional[torch.Tensor] = None, attn_res: Optional[torch.Tensor] = None, batch=None,alpha=0.5):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            attn_res: Attention residual mask
            batch: batch size
        
        Return:
            x: Output tensor
            attn_res: Attention residual for the upcomming transformer layer
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + self._get_rel_pos_bias()

        #Attention propagation module
        if attn_res is not None:
            attn_res = attn_res.reshape(batch, 1, -1, attn_res.shape[-2], attn_res.shape[-1])
            shape = attn.shape
            attn = attn.reshape(batch, 1, -1, attn.shape[-2], attn.shape[-1])
            attn_res = torch.nn.functional.interpolate(attn_res, (attn.shape[-3], shape[-2], shape[-1]))

            # if attn_res.shape[1] != attn.shape[1]:
            #     attn_res = attn_res.mean(1, keepdim=True)
        

            # attn = (0.5 * attn) + (0.5 * attn_res)
            # 残差权重
            attn = (alpha * attn) + ((1-alpha) * attn_res)   
            attn = attn.view(shape)
        attn_res = attn    

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_res


try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

# mlp层
class Mlp(nn.Module):
    #定义了基础的多层感知机
    def __init__(self, in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 # GELU激活函数 
                  act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()# 报错
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# 基础层实体
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 window_size=7, 
                 shift_size=0,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 head_dim = None,
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU,  
                 norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, 
            num_heads=num_heads, 
            head_dim=head_dim, 
            window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop
            alpha = 0.5)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process
        
        #todo 注意力插件 用于代替FNN
        self.CSA = CSA(
                channels =  dim,
                timeWindows= 1, 
                c_ratio=16, 
                t_ratio=1
            )
        
        
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                # 滚动张量，让每一个切割的窗口都能够获取特征信息
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows 分割张量为窗口数量
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows,attn_res = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # todo 用CSA代替FFN
        x = x+self.CSA(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

# 做了一个线性变化层,维度映射
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks 创建多个swin-transformer block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            # 根据模型深度参数创建不同的层数实例
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    # 用于计算该类的计算损耗
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

# 图像转化为patches ，通过一个2D卷积实现，并且涉及到转换操作
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        #定义卷积层
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.proj = Snn_Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 是否规范化
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # 异常处理
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # 次数输出的张量为[B Ph*Pw C]的三维张量，高度和宽度被展平了，并且和C替换维度顺序了
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

# 调用模型
class model_Test_Res_MS_swin(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    # 模型入口
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches 图像处理 
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        
        # 设置完模型参数后,对模型进行训练正则化的设置

        # absolute position embedding 绝对位置嵌入
        if self.ape:
            # 将absolute_pos_embed设置为一个可以梯度更新的参数,并设置基础的张量形状
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            #初始化
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # Dropout 正则化
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers 
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               # 在符合条件的情况下,做一次线性变化
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # MLP分类 
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
                
    #返回不会衰减的权重字典  
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    #返回不会被衰减的关键词字典
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x): # 被第一个调用的方法
        # 提取图像特征
        x = self.patch_embed(x) 
        # 是否绝对位置嵌入
        if self.ape:
            # ？加法？怎么感觉像一个残差连接
            # 这确实类似残差连接的设计，将一个超参数纳入反向传播，通过与特征值相加纳入计算
            x = x + self.absolute_pos_embed
        # 
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
