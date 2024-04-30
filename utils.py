import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import functools
import os, numbers, math, signal
from typing import Union, List, Sequence
from distutils.version import LooseVersion
import warnings
import torch as th

warnings.filterwarnings("ignore")
EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class ComplexTensor:
    def __init__(self, real: Union[torch.Tensor, np.ndarray], imag=None):
        if imag is None:
            if isinstance(real, np.ndarray):
                if real.dtype.kind == 'c':
                    imag = real.imag
                    real = real.real
                else:
                    imag = np.zeros_like(real)
            else:
                imag = torch.zeros_like(real)

        if isinstance(real, np.ndarray):
            real = torch.from_numpy(real)
        if isinstance(imag, np.ndarray):
            imag = torch.from_numpy(imag)

        if not torch.is_tensor(real):
            raise TypeError(
                f'The first arg must be torch.Tensor'
                f'but got {type(real)}')

        if not torch.is_tensor(imag):
            raise TypeError(
                f'The second arg must be torch.Tensor'
                f'but got {type(imag)}')
        if not real.size() == imag.size():
            raise ValueError(f'The two inputs must have same sizes: '
                             f'{real.size()} != {imag.size()}')

        self.real = real
        self.imag = imag

    def __getitem__(self, item) -> 'ComplexTensor':
        return ComplexTensor(self.real[item], self.imag[item])

    def __setitem__(self, item, value: Union['ComplexTensor',
                                             torch.Tensor, numbers.Number]):
        if isinstance(value, (ComplexTensor, complex)):
            self.real[item] = value.real
            self.imag[item] = value.imag
        else:
            self.real[item] = value
            self.imag[item] = 0

    def __mul__(self,
                other: Union['ComplexTensor', torch.Tensor, numbers.Number]) \
            -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real)
        else:
            return ComplexTensor(self.real * other, self.imag * other)

    def __rmul__(self,
                 other: Union['ComplexTensor', torch.Tensor, numbers.Number]) \
            -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(
                other.real * self.real - other.imag * self.imag,
                other.imag * self.real + other.real * self.imag)
        else:
            return ComplexTensor(other * self.real, other * self.imag)

    def __imul__(self, other):
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self * other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real *= other
            self.imag *= other
        return self

    def __truediv__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            den = other.real ** 2 + other.imag ** 2
            return ComplexTensor(
                (self.real * other.real + self.imag * other.imag) / den,
                (-self.real * other.imag + self.imag * other.real) / den)
        else:
            return ComplexTensor(self.real / other, self.imag / other)

    def __rtruediv__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            den = self.real ** 2 + self.imag ** 2
            return ComplexTensor(
                (other.real * self.real + other.imag * self.imag) / den,
                (-other.real * self.imag + other.imag * self.real) / den)
        else:
            den = self.real ** 2 + self.imag ** 2
            return ComplexTensor(other * self.real / den,
                                 -other * self.imag / den)

    def __itruediv__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self / other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real /= other
            self.imag /= other
        return self

    def __add__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(self.real + other.real,
                                 self.imag + other.imag)
        else:
            return ComplexTensor(self.real + other, self.imag)

    def __radd__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(other.real + self.real,
                                 other.imag + self.imag)
        else:
            return ComplexTensor(other + self.real, self.imag)

    def __iadd__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            self.real += other.real
            self.imag += other.imag
        else:
            self.real += other
        return self

    def __sub__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(self.real - other.real,
                                 self.imag - other.imag)
        else:
            return ComplexTensor(self.real - other, self.imag)

    def __rsub__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(other.real - self.real,
                                 other.imag - self.imag)
        else:
            return ComplexTensor(other - self.real, self.imag)

    def __isub__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, complex)):
            self.real -= other.real
            self.imag -= other.imag
        else:
            self.real -= other
        return self

    def __matmul__(self, other) -> 'ComplexTensor':
        if isinstance(other, ComplexTensor):
            o_real = torch.matmul(self.real, other.real) - \
                     torch.matmul(self.imag, other.imag)
            o_imag = torch.matmul(self.real, other.imag) + \
                     torch.matmul(self.imag, other.real)
        else:
            o_real = torch.matmul(self.real, other)
            o_imag = torch.matmul(self.imag, other)
        return ComplexTensor(o_real, o_imag)

    def __rmatmul__(self, other) -> 'ComplexTensor':
        if isinstance(other, ComplexTensor):
            o_real = torch.matmul(other.real, self.real) - \
                     torch.matmul(other.imag, self.imag)
            o_imag = torch.matmul(other.real, self.imag) + \
                     torch.matmul(other.imag, self.real)
        else:
            o_real = torch.matmul(other, self.real)
            o_imag = torch.matmul(other, self.imag)
        return ComplexTensor(o_real, o_imag)

    def __imatmul__(self, other) -> 'ComplexTensor':
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self @ other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real @= other
            self.imag @= other
        return self

    def __neg__(self) -> 'ComplexTensor':
        return ComplexTensor(-self.real, -self.imag)

    def __eq__(self, other) -> torch.Tensor:
        if isinstance(other, (ComplexTensor, complex)):
            return (self.real == other.real) ** (self.imag == other.imag)
        else:
            return (self.real == other) ** (self.imag == 0)

    def __len__(self) -> int:
        return len(self.real)

    def __repr__(self) -> str:
        return 'ComplexTensor(\nReal:\n' \
               + repr(self.real) + '\nImag:\n' + repr(self.imag) + '\n)'

    def __abs__(self) -> torch.Tensor:
        return (self.real * self.real + self.imag * self.imag).sqrt()

    def __pow__(self, exponent) -> 'ComplexTensor':
        if exponent == -2:
            return 1 / (self * self)
        if exponent == -1:
            return 1 / self
        if exponent == 0:
            return ComplexTensor(torch.ones_like(self.real))
        if exponent == 1:
            return self.clone()
        if exponent == 2:
            return self * self

        _abs = self.abs().pow(exponent)
        _angle = exponent * self.angle()
        return ComplexTensor(_abs * torch.cos(_angle),
                             _abs * torch.sin(_angle))

    def __ipow__(self, exponent) -> 'ComplexTensor':
        c = self ** exponent
        self.real = c.real
        self.imag = c.imag
        return self

    def abs(self) -> torch.Tensor:
        return (self.real * self.real + self.imag * self.imag).sqrt()

    def angle(self) -> torch.Tensor:
        return torch.atan2(self.imag, self.real)

    def backward(self) -> None:
        self.real.backward()
        self.imag.backward()

    def byte(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.byte(), self.imag.byte())

    def clone(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.clone(), self.imag.clone())

    def flatten(self, dim) -> 'ComplexTensor':
        return ComplexTensor(torch.flatten(self.real, dim), torch.flatten(self.imag, dim))

    def reshape(self, *shape) -> 'ComplexTensor':
        return ComplexTensor(self.real.reshape(shape), self.imag.reshape(shape))

    def zeromean(self, dim) -> 'ComplexTensor':
        self.real = self.real - torch.mean(self.real, dim=dim, keepdim=True)
        self.imag = self.imag - torch.mean(self.imag, dim=dim, keepdim=True)
        return ComplexTensor(self.real, self.imag)

    def conj(self) -> 'ComplexTensor':
        return ComplexTensor(self.real, -self.imag)

    def conj_(self) -> 'ComplexTensor':
        self.imag.neg_()
        return self

    def contiguous(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.contiguous(), self.imag.contiguous())

    def copy_(self) -> 'ComplexTensor':
        self.real = self.real.copy_()
        self.imag = self.imag.copy_()
        return self

    def cpu(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.cpu(), self.imag.cpu())

    def cuda(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.cuda(), self.imag.cuda())

    def expand(self, *sizes):
        return ComplexTensor(self.real.expand(*sizes),
                             self.imag.expand(*sizes))

    def expand_as(self, *args, **kwargs):
        return ComplexTensor(self.real.expand_as(*args, **kwargs),
                             self.imag.expand_as(*args, **kwargs))

    def detach(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.detach(), self.imag.detach())

    def detach_(self) -> 'ComplexTensor':
        self.real.detach_()
        self.imag.detach_()
        return self

    @property
    def device(self):
        assert self.real.device == self.imag.device
        return self.real.device

    def diag(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.diag(), self.imag.diag())

    def diagonal(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.diag(), self.imag.diag())

    def dim(self) -> int:
        return self.real.dim()

    def double(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.double(), self.imag.double())

    @property
    def dtype(self) -> torch.dtype:
        return self.real.dtype

    def eq(self, other) -> torch.Tensor:
        if isinstance(other, (ComplexTensor, complex)):
            return (self.real == other.real) * (self.imag == other.imag)
        else:
            return (self.real == other) * (self.imag == 0)

    def equal(self, other) -> bool:
        if isinstance(other, (ComplexTensor, complex)):
            return self.real.equal(other.real) and self.imag.equal(other.imag)
        else:
            return self.real.equal(other) and self.imag.equal(0)

    def float(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.float(), self.imag.float())

    def fill(self, value) -> 'ComplexTensor':
        if isinstance(value, complex):
            return ComplexTensor(self.real.fill(value.real),
                                 self.imag.fill(value.imag))
        else:
            return ComplexTensor(self.real.fill(value), self.imag.fill(0))

    def fill_(self, value) -> 'ComplexTensor':
        if isinstance(value, complex):
            self.real.fill_(value.real)
            self.imag.fill_(value.imag)
        else:
            self.real.fill_(value)
            self.imag.fill_(0)
        return self

    def gather(self, dim, index) -> 'ComplexTensor':
        return ComplexTensor(self.real.gather(dim, index),
                             self.real.gather(dim, index))

    def get_device(self, *args, **kwargs):
        return self.real.get_device(*args, **kwargs)

    def half(self) -> 'ComplexTensor':
        return ComplexTensor(self.real.half(), self.imag.half())

    def index_add(self, dim, index, tensor) -> 'ComplexTensor':
        return ComplexTensor(self.real.index_add(dim, index, tensor),
                             self.imag.index_add(dim, index, tensor))

    def index_copy(self, dim, index, tensor) -> 'ComplexTensor':
        return ComplexTensor(self.real.index_copy(dim, index, tensor),
                             self.imag.index_copy(dim, index, tensor))

    def index_fill(self, dim, index, value) -> 'ComplexTensor':
        return ComplexTensor(self.real.index_fill(dim, index, value),
                             self.imag.index_fill(dim, index, value))

    def index_select(self, dim, index) -> 'ComplexTensor':
        return ComplexTensor(self.real.index_select(dim, index),
                             self.imag.index_select(dim, index))

    def inverse(self, ntry=5):
        # m x n x n
        in_size = self.size()
        a = self.view(-1, self.size(-1), self.size(-1))
        # see "The Matrix Cookbook" (http://www2.imm.dtu.dk/pubdb/p.php?3274)
        # "Section 4.3"
        for i in range(ntry):
            t = i * 0.1

            e = a.real + t * a.imag
            f = a.imag - t * a.real

            try:
                x = torch.matmul(f, e.inverse())
                z = (e + torch.matmul(x, f)).inverse()
            except Exception:
                if i == ntry - 1:
                    raise
                continue

            if t != 0.:
                eye = torch.eye(a.real.size(-1), dtype=a.real.dtype,
                                device=a.real.device)[None]
                o_real = torch.matmul(z, (eye - t * x))
                o_imag = -torch.matmul(z, (t * eye + x))
            else:
                o_real = z
                o_imag = -torch.matmul(z, x)

            o = ComplexTensor(o_real, o_imag)
            return o.view(*in_size)

    def item(self) -> numbers.Number:
        return self.real.item() + 1j * self.imag.item()

    def masked_fill(self, mask, value) -> 'ComplexTensor':
        if isinstance(value, complex):
            return ComplexTensor(self.real.masked_fill(mask, value.real),
                                 self.imag.masked_fill(mask, value.imag))

        else:
            return ComplexTensor(self.real.masked_fill(mask, value),
                                 self.imag.masked_fill(mask, 0))

    def masked_fill_(self, mask, value) -> 'ComplexTensor':
        if isinstance(value, complex):
            self.real.masked_fill_(mask, value.real)
            self.imag.masked_fill_(mask, value.imag)
        else:
            self.real.masked_fill_(mask, value)
            self.imag.masked_fill_(mask, 0)
        return self

    def mean(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.mean(*args, **kwargs),
                             self.imag.mean(*args, **kwargs))

    def neg(self) -> 'ComplexTensor':
        return ComplexTensor(-self.real, -self.imag)

    def neg_(self) -> 'ComplexTensor':
        self.real.neg_()
        self.imag.neg_()
        return self

    def nelement(self) -> int:
        return self.real.nelement()

    def numel(self) -> int:
        return self.real.numel()

    def new(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.new(*args, **kwargs),
                             self.imag.new(*args, **kwargs))

    def new_empty(self, size, dtype=None, device=None, requires_grad=False) \
            -> 'ComplexTensor':
        real = self.real.new_empty(size,
                                   dtype=dtype,
                                   device=device,
                                   requires_grad=requires_grad)
        imag = self.imag.new_empty(size,
                                   dtype=dtype,
                                   device=device,
                                   requires_grad=requires_grad)
        return ComplexTensor(real, imag)

    def new_full(self, size, fill_value, dtype=None, device=None,
                 requires_grad=False) -> 'ComplexTensor':
        if isinstance(fill_value, complex):
            real_value = fill_value.real
            imag_value = fill_value.imag
        else:
            real_value = fill_value
            imag_value = 0.

        real = self.real.new_full(size,
                                  fill_value=real_value,
                                  dtype=dtype,
                                  device=device,
                                  requires_grad=requires_grad)
        imag = self.imag.new_full(size,
                                  fill_value=imag_value,
                                  dtype=dtype,
                                  device=device,
                                  requires_grad=requires_grad)
        return ComplexTensor(real, imag)

    def new_tensor(self, data, dtype=None, device=None,
                   requires_grad=False) -> 'ComplexTensor':
        if isinstance(data, ComplexTensor):
            real = data.real
            imag = data.imag
        elif isinstance(data, np.ndarray):
            if data.dtype.kind == 'c':
                real = data.real
                imag = data.imag
            else:
                real = data
                imag = None
        else:
            real = data
            imag = None

        real = self.real.new_tensor(real,
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
        if imag is None:
            imag = torch.zeros_like(real,
                                    dtype=dtype,
                                    device=device,
                                    requires_grad=requires_grad)
        else:
            imag = self.imag.new_tensor(imag,
                                        dtype=dtype,
                                        device=device,
                                        requires_grad=requires_grad)
        return ComplexTensor(real, imag)

    def numpy(self) -> np.ndarray:
        return self.real.numpy() + 1j * self.imag.numpy()

    def permute(self, *dims) -> 'ComplexTensor':
        return ComplexTensor(self.real.permute(*dims),
                             self.imag.permute(*dims))

    def pow(self, exponent) -> 'ComplexTensor':
        return self ** exponent

    def requires_grad_(self) -> 'ComplexTensor':
        self.real.requires_grad_()
        self.imag.requires_grad_()
        return self

    @property
    def requires_grad(self):
        assert self.real.requires_grad == self.imag.requires_grad
        return self.real.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.real.requires_grad = value
        self.imag.requires_grad = value

    def repeat(self, *sizes):
        return ComplexTensor(self.real.repeat(*sizes),
                             self.imag.repeat(*sizes))

    def retain_grad(self) -> 'ComplexTensor':
        self.real.retain_grad()
        self.imag.retain_grad()
        return self

    def share_memory_(self) -> 'ComplexTensor':
        self.real.share_memory_()
        self.imag.share_memory_()
        return self

    @property
    def shape(self) -> torch.Size:
        return self.real.shape

    def size(self, *args, **kwargs) -> torch.Size:
        return self.real.size(*args, **kwargs)

    def sqrt(self) -> 'ComplexTensor':
        return self ** 0.5

    def squeeze(self, dim) -> 'ComplexTensor':
        return ComplexTensor(self.real.squeeze(dim),
                             self.imag.squeeze(dim))

    def sum(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.sum(*args, **kwargs),
                             self.imag.sum(*args, **kwargs), )

    def take(self, indices) -> 'ComplexTensor':
        return ComplexTensor(self.real.take(indices), self.imag.take(indices))

    def to(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.to(*args, **kwargs),
                             self.imag.to(*args, **kwargs))

    def tolist(self) -> List[numbers.Number]:
        return [r + 1j * i
                for r, i in zip(self.real.tolist(), self.imag.tolist())]

    def transpose(self, dim0, dim1) -> 'ComplexTensor':
        return ComplexTensor(self.real.transpose(dim0, dim1),
                             self.imag.transpose(dim0, dim1))

    def transpose_(self, dim0, dim1) -> 'ComplexTensor':
        self.real.transpose_(dim0, dim1)
        self.imag.transpose_(dim0, dim1)
        return self

    def type(self) -> str:
        return self.real.type()

    def unfold(self, dim, size, step):
        return ComplexTensor(self.real.unfold(dim, size, step),
                             self.imag.unfold(dim, size, step))

    def unsqueeze(self, dim) -> 'ComplexTensor':
        return ComplexTensor(self.real.unsqueeze(dim),
                             self.imag.unsqueeze(dim))

    def unsqueeze_(self, dim) -> 'ComplexTensor':
        self.real.unsqueeze_(dim)
        self.imag.unsqueeze_(dim)
        return self

    def view(self, *args, **kwargs) -> 'ComplexTensor':
        return ComplexTensor(self.real.view(*args, **kwargs),
                             self.imag.view(*args, **kwargs))

    def view_as(self, tensor):
        return self.view(tensor.size())


def init_kernel(frame_len, frame_hop, num_fft=None, window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    fft_size = 2 ** math.ceil(math.log2(frame_len)) if not num_fft else num_fft
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    w = torch.fft.rfft(torch.eye(fft_size) / S_)
    kernel = torch.stack([w.real, w.imag], -1)
    kernel = torch.transpose(kernel, 0, 2) * window
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class STFTBase(nn.Module):
    def __init__(self, frame_len, frame_hop, window="sqrt_hann", num_fft=None):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len, frame_hop, num_fft=num_fft, window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self): self.K.requires_grad = False

    def unfreeze(self): self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))

    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(self.window, self.stride,
                                                                                             self.K.requires_grad,
                                                                                             self.K.shape)


class STFT(STFTBase):
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            print(x.shape)
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i


class iSTFT(STFTBase):
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        c = torch.cat([r, i], dim=1)
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s


class Conv1D(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self._name = 'Identity'

    def forward(self, x): return x


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


def _fcomplex(func, nthargs=0):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Union[ComplexTensor, torch.Tensor]:
        signal = args[nthargs]
        if isinstance(signal, ComplexTensor):
            real_args = args[:nthargs] + (signal.real,) + args[nthargs + 1:]
            imag_args = args[:nthargs] + (signal.imag,) + args[nthargs + 1:]
            real = func(*real_args, **kwargs)
            imag = func(*imag_args, **kwargs)
            return ComplexTensor(real, imag)
        else:
            return func(*args, **kwargs)

    return wrapper


def einsum(equation, operands):
    """Einsum

    >>> import numpy
    >>> def get(*shape):
    ...     real = np.random.rand(*shape)
    ...     imag = np.random.rand(*shape)
    ...     return real + 1j * imag
    >>> x = get(3, 4, 5)
    >>> y = get(3, 5, 6)
    >>> z = get(3, 6, 7)
    >>> test = einsum('aij,ajk,akl->ail',
    ...               [ComplexTensor(x), ComplexTensor(y), ComplexTensor(z)])
    >>> valid = np.einsum('aij,ajk,akl->ail', x, y, z)
    >>> np.testing.assert_allclose(test.numpy(), valid)

    """
    x = operands[0]
    if isinstance(x, ComplexTensor):
        real_operands = [[x.real]]
        imag_operands = [[x.imag]]
    else:
        real_operands = [[x]]
        imag_operands = []

    for x in operands[1:]:
        if isinstance(x, ComplexTensor):
            real_operands, imag_operands = \
                [ops + [x.real] for ops in real_operands] + \
                [ops + [-x.imag] for ops in imag_operands], \
                [ops + [x.imag] for ops in real_operands] + \
                [ops + [x.real] for ops in imag_operands]
        else:
            real_operands = [ops + [x] for ops in real_operands]
            imag_operands = [ops + [x] for ops in imag_operands]

    real = sum([torch.einsum(equation, ops) for ops in real_operands])
    imag = sum([torch.einsum(equation, ops) for ops in imag_operands])
    return ComplexTensor(real, imag)


def cat(seq: Sequence[Union[ComplexTensor, torch.Tensor]], dim=0, out=None):
    reals = [v.real if isinstance(v, ComplexTensor) else v for v in seq]
    imags = [v.imag if isinstance(v, ComplexTensor)
             else torch.zeros_like(v.real) for v in seq]
    if out is not None:
        out_real = out.real
        out_imag = out.imag
    else:
        out_real = out_imag = None
    return ComplexTensor(torch.cat(reals, dim, out=out_real),
                         torch.cat(imags, dim, out=out_imag))


def stack(seq: Sequence[Union[ComplexTensor, torch.Tensor]], dim=0, out=None):
    reals = [v.real if isinstance(v, ComplexTensor) else v for v in seq]
    imags = [v.imag if isinstance(v, ComplexTensor)
             else torch.zeros_like(v.real) for v in seq]
    if out is not None:
        out_real = out.real
        out_imag = out.imag
    else:
        out_real = out_imag = None
    return ComplexTensor(torch.stack(reals, dim, out=out_real),
                         torch.stack(imags, dim, out=out_imag))


pad = _fcomplex(F.pad)


@_fcomplex
def reverse(tensor: torch.Tensor, dim=0) -> torch.Tensor:
    # https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
    idx = [i for i in range(tensor.size(dim) - 1, -1, -1)]
    idx = torch.LongTensor(idx).to(tensor.device)
    inverted_tensor = tensor.index_select(dim, idx)
    return inverted_tensor


@_fcomplex
def signal_frame(signal: torch.Tensor,
                 frame_length: int, frame_step: int,
                 pad_value=0) -> torch.Tensor:
    """Expands signal into frames of frame_length.

    Args:
        signal : (B * F, D, T)
    Returns:
        torch.Tensor: (B * F, D, T, W)
    """
    signal = F.pad(signal, (0, frame_length - 1), 'constant', pad_value)
    indices = sum([list(range(i, i + frame_length))
                   for i in range(0, signal.size(-1) - frame_length + 1,
                                  frame_step)], [])

    signal = signal[..., indices].view(*signal.size()[:-1], -1, frame_length)
    return signal


def trace(a: ComplexTensor) -> ComplexTensor:
    if LooseVersion(torch.__version__) >= LooseVersion('1.3'):
        datatype = torch.bool
    else:
        datatype = torch.uint8
    E = torch.eye(a.real.size(-1), dtype=datatype).expand(*a.size())
    if LooseVersion(torch.__version__) >= LooseVersion('1.1'):
        E = E.type(torch.bool)
    return a[E].view(*a.size()[:-1]).sum(-1)


def allclose(a: Union[ComplexTensor, torch.Tensor],
             b: Union[ComplexTensor, torch.Tensor],
             rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return torch.allclose(a.real, b.real,
                              rtol=rtol, atol=atol, equal_nan=equal_nan) and \
               torch.allclose(a.imag, b.imag,
                              rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif not isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return torch.allclose(a.real, b.real,
                              rtol=rtol, atol=atol, equal_nan=equal_nan) and \
               torch.allclose(torch.zeros_like(b.imag), b.imag,
                              rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif isinstance(a, ComplexTensor) and not isinstance(b, ComplexTensor):
        return torch.allclose(a.real, b,
                              rtol=rtol, atol=atol, equal_nan=equal_nan) and \
               torch.allclose(a.imag, torch.zeros_like(a.imag),
                              rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def matmul(a: Union[ComplexTensor, torch.Tensor],
           b: Union[ComplexTensor, torch.Tensor]) -> ComplexTensor:
    if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return a @ b
    elif not isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        o_imag = torch.matmul(a, b.imag)
    elif isinstance(a, ComplexTensor) and not isinstance(b, ComplexTensor):
        return a @ b
    else:
        o_real = torch.matmul(a.real, b.real)
        o_imag = torch.zeros_like(o_real)
    return ComplexTensor(o_real, o_imag)


def complex_matrix2real_matrix(c: ComplexTensor) -> torch.Tensor:
    # NOTE(kamo):
    # Complex value can be expressed as follows
    #   a + bi => a * x + b y
    # where
    #   x = |1 0|  y = |0 -1|
    #       |0 1|,     |1  0|
    # A complex matrix can be also expressed as
    #   |A -B|
    #   |B  A|
    # and complex vector can be expressed as
    #   |A|
    #   |B|

    assert c.size(-2) == c.size(-1), c.size()
    # (∗, m, m) -> (*, 2m, 2m)
    return torch.cat(
        [torch.cat([c.real, -c.imag], dim=-1), torch.cat([c.imag, c.real], dim=-1)],
        dim=-2,
    )


def complex_vector2real_vector(c: ComplexTensor) -> torch.Tensor:
    # (∗, m, k) -> (*, 2m, k)
    return torch.cat([c.real, c.imag], dim=-2)


def real_matrix2complex_matrix(c: torch.Tensor) -> ComplexTensor:
    assert c.size(-2) == c.size(-1), c.size()
    # (∗, 2m, 2m) -> (*, m, m)
    n = c.size(-1)
    assert n % 2 == 0, n
    real = c[..., : n // 2, : n // 2]
    imag = c[..., n // 2:, : n // 2]
    return ComplexTensor(real, imag)


def real_vector2complex_vector(c: torch.Tensor) -> ComplexTensor:
    # (∗, 2m, k) -> (*, m, k)
    n = c.size(-2)
    assert n % 2 == 0, n
    real = c[..., : n // 2, :]
    imag = c[..., n // 2:, :]
    return ComplexTensor(real, imag)


def solve(b: ComplexTensor, a: ComplexTensor) -> ComplexTensor:
    """Solve ax = b"""
    a = complex_matrix2real_matrix(a)
    b = complex_vector2real_vector(b)
    x, LU = torch.solve(b, a)

    return real_vector2complex_vector(x), real_matrix2complex_matrix(LU)
