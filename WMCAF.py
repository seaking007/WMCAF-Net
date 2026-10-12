
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
import numpy as np
from pytorch_wavelets import DWTForward
from network.vmamba import *
from Fmamba_network.LN.LN import *
from torch.nn.init import _calculate_fan_in_and_fan_out
from pytorch_wavelets import DWTInverse
from torch.autograd import Function


class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=(0, padding),
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=(1, 1))
        return out_normal


class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_Diag_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=(0, padding),
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.zeros(C_out, C_in, 1, device=x.device)
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=(1, 1))
        return out_normal

class Conv2d_3x3_Normal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_3x3_Normal, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)



class DoubleConv(nn.Module):
    def __init__(self, cin, cout, k=3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, padding=k // 2),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=k, padding=k // 2),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input=x

        x2= self.conv(input)

        return x2

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H=None, W=None):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class vss(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        # self.ffn = SMFFL(in_channels=hidden_dim)
    def forward(self, input: torch.Tensor):
        x = input+self.drop_path(self.self_attention(self.ln(input)))
        # residual = x
        # x = residual + self.drop_path(self.ffn(self.ln(x)))

        return x







class DWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 保证输入张量x在内存中是连续存储的
        x = x.contiguous()
        # 保存后向传播需要的参数
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 获取输入张量x的通道数
        dim = x.shape[1]
        # 对x进行二维卷积操作，得到低频和高频分量
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 将四个分量按通道维度拼接起来
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回拼接后的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的权重
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            # 调整dx的维度顺序并重塑
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            # 将四个小波滤波器沿零维度拼接，并重复C次以匹配输入通道数
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            # 使用转置卷积进行上采样
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        # 返回dx以及其余不需要梯度的参数
        return dx, None, None, None, None

# 定义一个二维离散小波变换模块，继承自nn.Module
class DWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建分解低通和高通滤波器的Tensor
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        # 计算二维分解滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        # 注册缓冲区变量来存储滤波器
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # 确保滤波器的数据类型为float32
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用DWT_Function的forward方法
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# 定义一个IDWT功能类，继承自Function
class IDWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, filters):
        # 保存后向传播需要的参数
        ctx.save_for_backward(filters)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 根据保存的形状信息调整x的形状
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        # 计算通道数
        C = x.shape[1]
        # 重塑x
        x = x.reshape(B, -1, H, W)
        # 重复滤波器C次以匹配输入通道数
        filters = filters.repeat(C, 1, 1, 1)
        # 使用转置卷积进行上采样
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        # 返回上采样的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的滤波器
            filters = ctx.saved_tensors
            filters = filters[0]
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            # 分解滤波器
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            # 对dx进行二维卷积操作，得到低频和高频分量
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            # 将四个分量按通道维度拼接起来
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回dx以及其余不需要梯度的参数
        return dx, None

# 定义一个二维逆离散小波变换模块，继承自nn.Module
class IDWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建重构低通和高通滤波器的Tensor
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        # 计算二维重构滤波器
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        # 为滤波器添加额外的维度
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        # 将四个小波滤波器沿零维度拼接
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        # 注册缓冲区变量来存储滤波器
        self.register_buffer('filters', filters)
        # 确保滤波器的数据类型为float32
        self.filters = self.filters.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用IDWT_Function的forward方法
        return IDWT_Function.apply(x, self.filters)


class DLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 初始化第一个深度可分离卷积层
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        # 初始化第二个深度可分离卷积层
        # self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        # self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)

        # 初始化空间注意力模块
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x1,x2):

        # 通过第一个卷积层
        att1 = self.att_conv1(x1)
        # 通过第二个卷积层
        att2 = self.att_conv2(x2)

        # 将两个卷积的结果在通道维度上拼接
        att = torch.cat([att1, att2], dim=1)
        # 计算平均注意力
        avg_att = torch.mean(att, dim=1, keepdim=True)
        # 计算最大注意力
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        # 将平均和最大注意力拼接
        att = torch.cat([avg_att, max_att], dim=1)
        # 通过空间注意力模块
        att = self.spatial_se(att)
        # print(att.size())
        # 计算输出，融合注意力
        output1 = att1 * att[:, 0, :, :].unsqueeze(1)
        output2 = att2 * att[:, 1, :, :].unsqueeze(1)
        # print(output1.size())
        # 将输入加回输出
        output = output1+output2 + x1 + x2
        return output


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)  # B 6C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


# dkl空间+FCM_通道
class CAFM(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(CAFM, self).__init__()
        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        # self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.dkl = DLK(dim=dim)
        self.apply(self._init_weights)
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)  # 1x1卷积用于融合


    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        weights = nn.ReLU()(self.weights)
        channel_weights = self.channel_weights(x1, x2)
        x1_1 = x1 + channel_weights[1] * x2
        x2_1 = x2 +  channel_weights[0] * x1
        out=self.dkl(x1_1,x2_1)
        return out


class WaveMamhba(nn.Module):
    def __init__(self, ch_in, ch_out, downsample=None):
        super(WaveMamhba, self).__init__()
        # …（前面代码保持不动）…
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

        self.vss = vss(hidden_dim=ch_in, drop_path=0.1, attn_drop_rate=0.1)


        self.down_dim = nn.Sequential(
            nn.Conv2d(ch_in * 2, ch_in, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.up_dim = nn.Sequential(
            nn.Conv2d(ch_in , ch_in*3, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in*3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.prj = nn.Conv2d(ch_in, ch_out, 1)
        self.ln = LayerNorm2d(ch_in)
        self.maxpool = nn.MaxPool2d(2)

        self.HV = Conv2d_Hori_Veri_Cross(ch_in, ch_in)
        self.DC = Conv2d_Diag_Cross(ch_in, ch_in)

        self.mlp = DWConv(ch_in)
        self.bn_re = nn.Sequential(
            nn.BatchNorm2d(ch_in ),
            nn.ReLU(inplace=True)
        )
        # 新增/替换的部分
        # 1. 水平+垂直高频的降维
        self.down_HV = nn.Sequential(
            nn.Conv2d(ch_in * 2, ch_in, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

        # 2. 斜高频的降维（可选，也可直接用原来的 down_dim，这里保持一致）
        self.down_diag = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

        # 3. 融合后的二次降维
        self.down_merge = nn.Sequential(
            nn.Conv2d(ch_in * 2, ch_in, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

        self.up_dim = nn.Sequential(
            nn.Conv2d(ch_in, ch_in * 3, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in * 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.DWconv_3_3 = nn.Sequential(
            nn.Conv2d(ch_in , ch_in, kernel_size=3, padding=1, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_in, kernel_size=1, bias=False),

        )
        # …（其余层不变）…

    def forward(self, x):
        input = x
        res = x

        x_w = self.dwt(input)
        x_ll, x_lh, x_hl, x_hh = x_w.chunk(4, dim=1)

        # 低频分支
        yL = x_ll
        b1 = self.ln(yL)
        b1 = b1.permute(0, 2, 3, 1)
        b1 = self.vss(b1)
        b1 = b1.permute(0, 3, 1, 2)

        # ========= 高频分支修改开始 =========
        # 1. 水平+垂直高频：拼接 → 1×1降维 → 十字卷积
        hv = torch.cat([x_lh, x_hl], dim=1)  # [B, 2*ch_in, H/2, W/2]
        hv = self.down_HV(hv)  # [B, ch_in, H/2, W/2]
        hv = self.HV(hv)  # 十字卷积

        # 2. 斜高频：对角卷积 → 降维
        diag = self.DC(x_hh)  # [B, ch_in, H/2, W/2]

        # 3. 融合：拼接 → 1×1降维
        b2 = torch.cat([hv, diag], dim=1)  # [B, 2*ch_in, H/2, W/2]
        b2 = self.down_merge(b2)  # [B, ch_in, H/2, W/2]
        # ========= 高频分支修改结束 =========

        # 上采样
        b2 = self.up_dim(b2)                         # [B, 3*ch_in, H/2, W/2]
        b = self.idwt(torch.cat([b1, b2], dim=1))    # [B, ch_in, H, W]

        # 残差 + MLP ……（后续保持不变）
        b = b + res

        input2 = b
        b = self.ln(b)
        b = self.mlp(b)
        b = b + input2
        b = self.prj(b)


        return b



class WMCAF(nn.Module):
    def __init__(self, num_classes=1,
                 input_channels=3
                 ):
        super().__init__()

        dims = [32, 64, 128, 256, 512]

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.e1 = nn.Sequential(
            DoubleConv(input_channels, dims[0]),
        )
        self.e2 = nn.Sequential(
            DoubleConv(dims[0], dims[1]),

        )
        self.e3 = nn.Sequential(
            DoubleConv(dims[1], dims[2]),

        )
        self.e4 = nn.Sequential(
            WaveMamhba(dims[2], dims[3], downsample=2),

        )
        self.e5 = nn.Sequential(
            WaveMamhba(dims[3], dims[4], downsample=2),
        )
        #
        self.d5 = nn.Sequential(
            WaveMamhba(dims[4], dims[3]),

        )
        self.d4 = nn.Sequential(
            WaveMamhba(dims[3], dims[2]),

        )
        self.d3 = nn.Sequential(
            DoubleConv(dims[2], dims[1]),

        )
        self.d2 = nn.Sequential(
            DoubleConv(dims[1], dims[0]),

        )
        self.d1 = nn.Sequential(
            DoubleConv(dims[0], dims[0] // 2),

        )
        self.out = nn.Conv2d(dims[0] // 2, num_classes, 1)
        self.skip4 = CAFM(dims[3])
        self.skip3 = CAFM(dims[2])
        self.skip2 = CAFM(dims[1])
        self.skip1 = CAFM(dims[0])


    def forward(self, x):
        x1 = self.e1(self.maxpool(x))
        x2 = self.e2(self.maxpool(x1))
        x3 = self.e3(self.maxpool(x2))
        x4 = self.e4(self.maxpool(x3))
        x5 = self.e5(self.maxpool(x4))


        d5 = self.d5(x5)
        d5 = self.skip4(self.upsample(d5), x4)

        d4 = self.d4(d5)
        d4 = self.skip3(self.upsample(d4), x3)

        d3 = self.d3(d4)
        d3 = self.skip2(self.upsample(d3), x2)

        d2 = self.d2(d3)
        d2 = self.skip1(self.upsample(d2), x1)

        d1 = self.d1(d2)
        d1 = self.upsample(d1)
        out = self.out(d1)



        return out


if __name__ == '__main__':
    ras = WMCAF(input_channels=3).cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output = ras(input_tensor)

    from thop import profile

    flops, params = profile(ras, (input_tensor,))
    print('Flops: %.2f G, Params: %.2f M' % (flops / 1e9, params / 1e6))
    # print(ras)

