import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from cplxmodule.nn import CplxConv2d, CplxConvTranspose2d, CplxBatchNorm2d, CplxLinear, CplxConv1d, CplxBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(9999)
EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


def param(nnet, Mb=True):
    neles = sum([param.nelement() for param in nnet.parameters()])
    return np.round(neles / 10 ** 6 if Mb else neles, 2)


# -----------------------   Architecture Parameters --------------------------------

# Step: 1.1 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Encoder/Decoder >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class ComplexEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False,
                 DSC=False):
        super(ComplexEncoder, self).__init__()
        # DSC: depthwise_separable_conv
        if DSC:
            self.conv = DSC_Encoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
        else:
            self.conv = CplxConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=bias)
        self.norm = CplxBatchNorm2d(out_channels)

    def forward(self, x):
        return complex_relu(self.norm(self.conv(x)))


class ComplexDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 output_padding=(0, 0), bias=False, DSC=False):
        super(ComplexDecoder, self).__init__()
        # DSC: depthwise_separable_conv
        if DSC:
            self.conv = DSC_Decoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    output_padding=output_padding, bias=bias)
        else:
            self.conv = CplxConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=output_padding, bias=bias)

        self.norm = CplxBatchNorm2d(out_channels)

    def forward(self, x):
        return complex_relu(self.norm(self.conv(x)))


class DSC_Encoder(nn.Module):
    # depthwise_separable_conv
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False):
        super(DSC_Encoder, self).__init__()
        self.depthwise = CplxConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    groups=in_channels, bias=bias)  # group = in_ch; and in_ch=out_ch
        self.pointwise = CplxConv2d(in_channels, out_channels, kernel_size=1, bias=bias)  # Kernel_size = 1 always

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DSC_Decoder(nn.Module):
    # depthwise_separable_conv
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=(1, 1), output_padding=(0, 0),
                 bias=False):
        super(DSC_Decoder, self).__init__()
        self.depthwise = CplxConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=in_channels, output_padding=output_padding,
                                             bias=bias)  # group = in_ch; and in_ch=out_ch
        self.pointwise = CplxConvTranspose2d(in_channels, out_channels, kernel_size=1,
                                             bias=bias)  # Kernel_size = 1 always

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# Step: 1.2 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Frequency Transformation Block >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class NodeReshape(nn.Module):
    def __init__(self, shape):
        super(NodeReshape, self).__init__()
        self.shape = shape

    def forward(self, feature_in: torch.Tensor):
        shape = feature_in.size()
        batch = shape[0]
        new_shape = [batch]
        new_shape.extend(list(self.shape))
        return feature_in.reshape(new_shape)


class Freq_FC(nn.Module):
    def __init__(self, F_dim, bias=False):
        super(Freq_FC, self).__init__()
        self.linear = CplxLinear(F_dim, F_dim, bias=bias)

    def forward(self, x):
        out = x.transpose(-1, -2).contiguous()  # [batch, channel_in_out, T, F]
        out = self.linear(out)  # .contiguous()
        out = torch.complex(out.real, out.imag)
        out = out.transpose(-1, -2).contiguous()  # [batch, channel_in_out, F, T]
        return out


class ComplexFTB(torch.nn.Module):
    """docstring for FTB"""

    def __init__(self, F_dim, channels):
        super(ComplexFTB, self).__init__()
        self.channels = channels
        self.C_r = 5
        self.F_dim = F_dim

        self.Conv2D_1 = nn.Sequential(
            CplxConv2d(in_channels=self.channels, out_channels=self.C_r, kernel_size=1, stride=1, padding=0),
            CplxBatchNorm2d(self.C_r),

        )
        self.Conv1D_1 = nn.Sequential(
            CplxConv1d(self.F_dim * self.C_r, self.F_dim, kernel_size=9, padding=4),
            CplxBatchNorm1d(self.F_dim),
        )
        self.FC = Freq_FC(self.F_dim, bias=False)
        self.Conv2D_2 = nn.Sequential(
            CplxConv2d(2 * self.channels, self.channels, kernel_size=1, stride=1, padding=0),
            CplxBatchNorm2d(self.channels),
        )

        self.att_inner_reshape = NodeReshape([self.F_dim * self.C_r, -1])
        self.att_out_reshape = NodeReshape([1, F_dim, -1])

    def cat(self, x, y, dim):
        real = torch.cat([x.real, y.real], dim)
        imag = torch.cat([x.imag, y.imag], dim)
        return ComplexTensor(real, imag)

    def forward(self, inputs, verbose=False):
        # feature_n: [batch, channel_in_out, T, F]

        _, _, self.F_dim, self.T_dim = inputs.shape
        # Conv2D
        out = complex_relu(self.Conv2D_1(inputs));
        if verbose: print('Layer-1               : ', out.shape)  # [B,Cr,T,F]
        # Reshape: [batch, channel_attention, F, T] -> [batch, channel_attention*F, T]
        out = out.view(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])
        # out = self.att_inner_reshape(out);
        if verbose: print('Layer-2               : ', out.shape)
        # out = out.view(-1, self.T_dim, self.F_dim * self.C_r) ; print(out.shape) # [B,c_ftb_r*f,segment_length]
        # Conv1D
        out = complex_relu(self.Conv1D_1(out));
        if verbose: print('Layer-3               : ', out.shape)  # [B,F, T]
        # temp = self.att_inner_reshape(temp); print(temp.shape)
        out = out.unsqueeze(1)
        # out = out.view(-1, self.channels, self.F_dim, self.T_dim);
        if verbose: print('Layer-4               : ', out.shape)  # [B,c_a,segment_length,1]
        # Multiplication with input
        out = out * inputs;
        if verbose: print('Layer-5               : ', out.shape)  # [B,c_a,segment_length,1]*[B,c_a,segment_length,f]
        # Frequency- FC
        # out = torch.transpose(out, 2, 3)  # [batch, channel_in_out, T, F]
        out = self.FC(out);
        # if verbose: print('Layer-6               : ', out.shape)  # [B,c_a,segment_length,f]
        # out = torch.transpose(out, 2, 3)  # [batch, channel_in_out, T, F]
        # Concatenation with Input
        out = self.cat(out, inputs, 1);
        if verbose: print('Layer-7               : ', out.shape)  # [B,2*c_a,segment_length,f]
        # Conv2D
        outputs = complex_relu(self.Conv2D_2(out));
        if verbose: print('Layer-8               : ', outputs.shape)  # [B,c_a,segment_length,f]

        return outputs


# -------------------------------- Depth wise Seperable Convolution --------------------------------
class depthwise_separable_convx(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_convx, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# Step: 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Skip Connection >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, DSC=False):
        super(SkipBlock, self).__init__()
        # DSC: depthwise_separable_conv

        if DSC:
            self.conv = DSC_Encoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=True)
        else:
            self.conv = CplxConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=True)

        self.norm = CplxBatchNorm2d(in_channels)

    def forward(self, x):
        return complex_relu(self.norm(self.conv(x))) + x


class SkipConnection(nn.Module):
    """
    SkipConnection is a concatenations of SkipBlocks
    """

    def __init__(self, in_channels, num_convblocks, DSC=False):
        super(SkipConnection, self).__init__()
        self.skip_blocks = [SkipBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, DSC=DSC) for k in
                            range(num_convblocks)]
        self.skip_path = nn.Sequential(*self.skip_blocks)

    def forward(self, x):
        return self.skip_path(x)


# Step: 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Activation Function >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def complex_mul(x, y):
    real = x.real * y.real - x.imag * y.imag
    imag = x.real * y.imag + x.imag * y.real
    out = ComplexTensor(real, imag)  # .contiguous()
    return out


def complex_sigmoid(input):
    return F.sigmoid(input.real).type(torch.complex64) + 1j * F.sigmoid(input.imag).type(torch.complex64)


class complex_softplus(nn.Module):
    def __init__(self):
        super(complex_softplus, self).__init__()
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, input):
        return self.softplus(input.real).type(torch.complex64) + 1j * self.softplus(input.imag).type(torch.complex64)


class complex_elu(nn.Module):
    def __init__(self):
        super(complex_elu, self).__init__()
        self.elu = nn.ELU(inplace=False)

    def forward(self, input):
        return self.elu(input.real).type(torch.complex64) + 1j * self.elu(input.imag).type(torch.complex64)


# Step: 4 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Bottleneck layers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step: 4.1 --------------------  GRU layers --------------------------
class ComplexGRU(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(ComplexGRU, self).__init__()
        self.rGRU = nn.Sequential(
            nn.GRU(input_size=input_size, hidden_size=input_size // 2, num_layers=num_layers, batch_first=True,
                   bidirectional=True), SelectItem(0))
        self.iGRU = nn.Sequential(
            nn.GRU(input_size=input_size, hidden_size=input_size // 2, num_layers=num_layers, batch_first=True,
                   bidirectional=True), SelectItem(0))
        self.linear = CplxLinear(input_size, output_size)

    def forward(self, x):
        x = x.transpose(-1, -2).contiguous()
        real = self.rGRU(x.real) - self.iGRU(x.imag)
        imag = self.rGRU(x.imag) + self.iGRU(x.real)
        out = self.linear(ComplexTensor(real, imag)).transpose(-1, -2)  # .contiguous()
        return out


# Step: 4.2 --------------------  Complex Transformer layers --------------------------


class Transformer_single(nn.Module):
    def __init__(self, nhead=8):
        super(Transformer_single, self).__init__()
        self.nhead = nhead

    def forward(self, x):
        # x = torch.randn(10, 2, 80, 256) [batch, Ch, F, T]
        b, c, F, T = x.shape
        STB = TransformerEncoderLayer(d_model=F, nhead=self.nhead)  # d_model = Expected feature
        STB.to("cuda")
        x = x.permute(1, 0, 3, 2).contiguous().view(-1, b * T, F)  # [c, b*T, F]
        x = x.to("cuda")
        x = STB(x)
        x = x.view(b, c, F, T)  # [b, c, F, T]
        return x


class Transformer_multi(nn.Module):
    # d_model = x.shape[3]
    def __init__(self, nhead, layer_num=2):
        super(Transformer_multi, self).__init__()
        self.layer_num = layer_num
        self.MTB = Transformer_single(nhead=nhead)  # d_model: the number of expected features in the input

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.MTB(x)
        return x


class ComplexTransformer(nn.Module):
    def __init__(self, nhead, num_layer):
        super(ComplexTransformer, self).__init__()
        self.rTrans = Transformer_multi(nhead=nhead, layer_num=num_layer)  # d_model = x.shape[3]
        self.iTrans = Transformer_multi(nhead=nhead, layer_num=num_layer)  # d_model = x.shape[3]
        # self.Trans = Transformer_multi(nhead=17, layer_num=num_layer)

    def forward(self, x):
        # real = self.Trans(x.real)
        # imag = self.Trans(x.imag)
        real = self.rTrans(x.real) - self.iTrans(x.imag)
        imag = self.rTrans(x.imag) + self.iTrans(x.real)
        out = ComplexTensor(real, imag)  # .contiguous()
        return out


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        # >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # >>> src = torch.rand(10, 32, 512)
        # >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).to("cuda")
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        if bidirectional:
            self.linear2 = Linear(d_model * 2 * 2, d_model)
        else:
            self.linear2 = Linear(d_model * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # src = src.to("cuda")
        # print("Tensor src evice:", src.device)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# Step: 4.3 --------------------  Complex DPRNN layers --------------------------

class ComplexDPAtBlock(nn.Module):
    def __init__(self, num_units, F, T, C):
        super(ComplexDPAtBlock, self).__init__()
        self.rTrans = DPAtBlock(num_units, F, T, C, causal=True)  # .cuda()
        self.iTrans = DPAtBlock(num_units, F, T, C, causal=True)  # .cuda()

    def forward(self, x):
        real_part = self.rTrans(x.real)
        img_part = self.iTrans(x.imag)
        real = real_part - img_part
        imag = real_part + img_part
        out = ComplexTensor(real, imag)
        return out


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, causal, attn_len):
        super(AttentionModel, self).__init__()
        self.causal = causal
        self.score = nn.Linear(hidden_size, hidden_size, bias=False)  # .cuda()
        self.attn_len = attn_len
        enhance_in = hidden_size * 2
        self.enhance = nn.Linear(enhance_in, hidden_size)  # .cuda()
        self.mask = nn.Linear(hidden_size, input_size)  # .cuda()

    def forward(self, k, q):
        k, q = k.cuda(), q.cuda()
        X1, X2, C = k.shape

        attn_score = torch.bmm(self.score(q), k.transpose(1, 2))  # .cuda()
        attn_max, _ = torch.max(attn_score, dim=-1, keepdim=True)
        exp_score = torch.exp(attn_score - attn_max)
        attn_weights = exp_score

        if self.causal:
            attn_weights = torch.tril(exp_score)
            if self.attn_len > 0:
                attn_weights = torch.triu(attn_weights, diagonal=-self.attn_len)

        weights_denom = torch.sum(attn_weights, dim=-1, keepdim=True)
        attn_weights = attn_weights / (weights_denom + 1e-30)
        c = torch.bmm(attn_weights, k)
        out = torch.cat((c, q), -1)
        out = self.enhance(out).tanh()
        mask_vector = self.mask(out).sigmoid()
        return mask_vector


class DPAtBlock(nn.Module):
    def __init__(self, num_units, F, T, C, causal, attn_len=0):
        # F= Fre axis dim, T= Time axis dim, C= Channel

        super(DPAtBlock, self).__init__()
        # Intra-segment attention components

        self.Intra_RNN_k_enc = nn.LSTM(input_size=num_units, hidden_size=num_units // 2, bidirectional=True,
                                       batch_first=True)
        self.Intra_RNN_q_enc = nn.LSTM(input_size=num_units, hidden_size=num_units // 2, bidirectional=True,
                                       batch_first=True)
        self.intra_attention = AttentionModel(input_size=num_units, hidden_size=num_units, causal=True, attn_len=0)

        # Inter-segment attention components
        self.Inter_RNN_k_enc = nn.LSTM(input_size=num_units, hidden_size=num_units, bidirectional=False,
                                       batch_first=True)
        self.Inter_RNN_q_enc = nn.LSTM(input_size=num_units, hidden_size=num_units, bidirectional=False,
                                       batch_first=True)
        self.inter_attention = AttentionModel(input_size=num_units, hidden_size=num_units, causal=True, attn_len=0)

        self.Intra_LN = nn.LayerNorm(normalized_shape=[F, C], elementwise_affine=True)
        self.Inter_LN = nn.LayerNorm(normalized_shape=[T, C], elementwise_affine=True)

    def forward(self, x):
        # Assuming x is of shape (batch_size, channel, freq, time)
        b, c, f, t = x.shape  # (bs,c,f,t)

        self.Intra_LN = nn.LayerNorm(normalized_shape=[f, c], elementwise_affine=True);
        self.Intra_LN = self.Intra_LN.to('cuda')
        self.Inter_LN = nn.LayerNorm(normalized_shape=[t, c], elementwise_affine=True);
        self.Inter_LN = self.Inter_LN.to('cuda')

        # Reshape for LSTM: (batch_size * channel * freq, time, input_dim)
        intra_inp = x.permute(0, 3, 2, 1).contiguous()  # (bs,T,F,C)
        x_reshaped = intra_inp.view(b * t, f, -1)  # (bs*T,F,C)
        # Intra-segment attention
        intra_normed = self.Intra_LN(x_reshaped)  # (bs*T,F,C)

        # Generate keys and queries for intra-segment attention using bi-directional LSTM
        k_intra, _ = self.Intra_RNN_k_enc(intra_normed)  # (bs*T,F,C)
        q_intra, _ = self.Intra_RNN_q_enc(intra_normed)  # (bs*T,F,C)

        # Apply intra-segment attention; note: we're using the normalized input x as the value here
        intra_mask = self.intra_attention(k_intra, q_intra)  # (bs*T,F,C)
        intra_mask = intra_mask.view(b, t, f, -1)
        intra_out = intra_inp * intra_mask  # (bs,T,F,C)

        # Inter-segment attention
        inter_LSTM_input = intra_out.permute(0, 2, 1, 3)  # (bs,F,T,C)
        inter_LSTM_input_reshaped = inter_LSTM_input.view(b * f, t, -1)  # (bs*F,T,C)
        inter_normed = self.Inter_LN(
            inter_LSTM_input_reshaped)  # Use the intra-segment output as input to the inter-segment attention

        # Generate keys and queries for inter-segment attention using uni-directional LSTM
        k_inter, _ = self.Inter_RNN_k_enc(inter_normed)  # (bs*F,T,C)
        q_inter, _ = self.Inter_RNN_q_enc(inter_normed)  # (bs*F,T,C)

        # Apply inter-segment attention
        inter_mask = self.inter_attention(q_inter, k_inter)  # (bs*F,T,C)
        inter_mask = inter_mask.view(b, f, t, -1)  # (bs,F,T,C)

        inter_out = inter_LSTM_input * inter_mask  # (bs,F,T,C)
        inter_out = inter_out.permute(0, 3, 1, 2)  # (bs,c,f,t)
        return inter_out


class ComplexDPRNNBlock(nn.Module):
    def __init__(self, numUnits, L, width, channel):
        super(ComplexDPRNNBlock, self).__init__()
        self.rTrans = DPRNNBlock(numUnits, L, width, channel, causal=True)  # d_model = x.shape[3]
        self.iTrans = DPRNNBlock(numUnits, L, width, channel, causal=True)  # d_model = x.shape[3]

    def forward(self, x):
        real = self.rTrans(x.real) - self.iTrans(x.imag)
        imag = self.rTrans(x.imag) + self.iTrans(x.real)
        out = ComplexTensor(real, imag)  # .contiguous()
        return out


class DPRNNBlock(nn.Module):
    def __init__(self, numUnits, L, width, channel, causal=True, **kwargs):
        super(DPRNNBlock, self).__init__(**kwargs)
        '''
        numUnits hidden layer size in the LSTM
        batch_size 
        L         number of frames, -1 for undefined length
        width     width size output from encoder
        channel   channel size output from encoder
        causal    instant Layer Norm or global Layer Norm
        '''
        self.numUnits = numUnits
        # self.batch_size = batch_size
        self.causal = causal
        self.L = L
        self.width = width
        self.channel = channel

        # -------------------  Intra ------------------------
        self.Intra_RNN = BidirectionalLSTM(num_units=self.numUnits)  # Bidirectional LSTM
        self.Intra_FC = DenseLayer(num_units=self.numUnits)  # FC layer

        # -------------------  Inter ------------------------
        self.Inter_RNN = LSTM(num_units=self.numUnits)  # Bidirectional LSTM
        self.Inter_FC = DenseLayer(num_units=self.numUnits)  # FC layer

        self.Intra_LN = nn.LayerNorm(normalized_shape=[self.width, self.channel], elementwise_affine=True)
        self.Inter_LN = nn.LayerNorm(normalized_shape=[1, self.channel], elementwise_affine=True)

    def forward(self, x):
        # self.Intra_LN = nn.LayerNorm(normalized_shape=[self.width, self.channel], elementwise_affine=True)
        # self.Intra_LN.to('cuda')
        # self.Inter_LN = nn.LayerNorm(normalized_shape=[1, self.channel], elementwise_affine=True)
        # self.Inter_LN.to('cuda')
        # --------------Intra-Chunk Processing-------------------
        # ++++++++++++++++++ BiLSTM +++++++++++++++++
        # change the x shape (bs,C,F,T) --> (bs,T,F,C)
        self.batch_size = x.shape[0]
        x = x.permute(0, 3, 2, 1)
        x.to('cuda')
        # input shape (bs,T,F,C) --> (bs*T,F,C)
        intra_LSTM_input = x.reshape(-1, self.width, self.channel)
        # (bs*T,F,C)
        intra_LSTM_out = self.Intra_RNN(intra_LSTM_input)
        # (bs*T,F,C) channel axis dense
        # ++++++++++++++++++ FC Layer +++++++++++++++++
        intra_dense_out = self.Intra_FC(intra_LSTM_out)
        # (bs*T,F,C) --> (bs,T,F,C) Freq and channel norm
        intra_ln_input = intra_dense_out.reshape(self.batch_size, -1, self.width, self.channel)
        # ++++++++++++++++++ Instant Length Normalization +++++++++++++++++
        if self.causal:
            # self.Intra_LN = nn.LayerNorm(normalized_shape=intra_ln_input.size()[2:], elementwise_affine=True)
            intra_out = self.Intra_LN(intra_ln_input)

        else:
            # (bs*T,F,C) --> (bs,T*F*C) global norm
            intra_ln_input = intra_dense_out.reshape(self.batch_size, -1)
            self.Intra_LN = nn.LayerNorm(normalized_shape=intra_ln_input.size()[0:], elementwise_affine=False)
            self.Intra_LN.to('cuda')
            intra_ln_out = self.Intra_LN(intra_ln_input)
            intra_out = intra_ln_out.reshape(self.batch_size, self.L, self.width, self.channel)  # (bs,T,F,C)
        intra_out = x + intra_out

        # --------------Inter-Chunk Processing-------------------

        # ++++++++++++++++++ LSTM +++++++++++++++++

        # (bs,T,F,C) --> (bs,F,T,C)
        inter_LSTM_input = intra_out.permute(0, 2, 1, 3)
        # (bs,F,T,C) --> (bs*F,T,C)
        inter_LSTM_input = inter_LSTM_input.reshape(self.batch_size * self.width, self.L, self.channel)
        inter_LSTM_out = self.Inter_RNN(inter_LSTM_input)

        # ++++++++++++++++++ FC Layer +++++++++++++++++
        # (bs,F,T,C)
        inter_dense_out = self.Inter_FC(inter_LSTM_out)
        inter_dense_out = inter_dense_out.reshape(self.batch_size, self.width, self.L, self.channel)

        # ++++++++++++++++++ Instant Length Normalization +++++++++++++++++

        if self.causal:
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_ln_input = inter_dense_out.permute(0, 2, 1, 3)
            # self.Inter_LN = nn.LayerNorm(normalized_shape=inter_ln_input.size()[2:], elementwise_affine=True)
            # self.Inter_LN.to('cuda')
            inter_out = self.Inter_LN(inter_ln_input)

        else:
            # (bs,F,T,C) --> (bs,F*T*C)
            inter_ln_input = inter_dense_out.reshape(self.batch_size, -1)
            self.Inter_LN = nn.LayerNorm(normalized_shape=inter_ln_input.size()[0:], elementwise_affine=False)
            self.Inter_LN.to('cuda')
            inter_ln_out = self.Inter_LN(inter_ln_input)

            inter_out = inter_ln_out.reshape(self.batch_size, self.width, self.L, self.channel)
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_out = inter_out.permute(0, 2, 1, 3)
        # (bs,T,F,C)
        inter_out = intra_out + inter_out
        # change to the input shape (bs,T,F,C) --> (bs,C,F,T)
        inter_out = inter_out.permute(0, 3, 2, 1)

        return inter_out


class BidirectionalLSTM(nn.Module):
    def __init__(self, num_units):
        super(BidirectionalLSTM, self).__init__()

        # Define a bidirectional LSTM layer
        # Since PyTorch LSTM inherently divides the hidden_size by 2 for bidirectional LSTMs,
        # we don't need to manually halve the number of units as in Keras.
        self.intra_rnn = nn.LSTM(input_size=num_units, hidden_size=num_units // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Pass the input through the bidirectional LSTM layer
        out, _ = self.intra_rnn(x)
        return out


class LSTM(nn.Module):
    def __init__(self, num_units):
        super(LSTM, self).__init__()

        # Define an LSTM layer
        self.inter_rnn = nn.LSTM(
            input_size=num_units,
            hidden_size=num_units,
            batch_first=True,
        )

    def forward(self, x):
        # Pass the input through the LSTM layer
        out, _ = self.inter_rnn(x)
        return out


class DenseLayer(nn.Module):
    def __init__(self, num_units):
        super(DenseLayer, self).__init__()

        # Define a Linear (Dense) layer
        self.intra_fc = nn.Linear(in_features=num_units, out_features=num_units)

    def forward(self, x):
        # Pass the input through the Linear (Dense) layer
        return self.intra_fc(x)


if __name__ == '__main__':

    # -------------------------  Test FTB block ------------------
    #
    x = 10 * torch.randn(10, 12, 128, 501)  # feature_n: [batch, channel_in, T, F] torch.Size([10, 12, 128, 501])
    x = torch.clamp(x, min=-1, max=1)
    model_FTB = ComplexFTB(F_dim=128, channels=x.shape[1]) #
    print("Input data shape: " + str(x.shape)) # Input data shape: torch.Size([10, 256, 512]) # feature_n: [batch, channel_in, T, F]
    y = model_FTB(x, verbose=True) # torch.Size([10, 12, 128, 501])
    print("Output data shape: " + str(y.shape)) # Output data shape: torch.Size([10, 256, 512]) # feature_n: [batch, channel_in, T, F]


