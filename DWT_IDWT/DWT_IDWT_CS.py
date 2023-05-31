"""
DWT and IDWT are implemented with existing convolution operations, their filter banks can be trainable parameters and initialize these filter banks with a given wavelet
The parameter trainable is set to False by default in each layer, which means that the filter bank is not updated by training; changing it to True means that the filter bank is updated.
If the dimension of the input data is smaller than kernel_size, an error is reported
Exact reconstruction at signal boundaries is not possible
These layers only support binary scalar wavelets for now. Other wavelets or superwavelets, such as a-binary wavelet, multiwavelet, wavelet frame, curved wavelet, ridge wavelet, strip wavelet, wavelet frame, etc. are not available at this time.
Currently, the boundary extension of 3D data does not have 'reflect', so even if we use haar wavelets, we cannot reconstruct accurately.
"""

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DWT_1D', 'IDWT_1D', 'DWT_2D', 'IDWT_2D', 'DWT_3D', 'IDWT_3D']
Pad_Mode = ['constant', 'reflect', 'replicate', 'circular']

class DWT_1D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        """
        :param pad_type: the boundary extension of the input data, theoretically using symmetric wavelets such as bior2.2\bior3.3, while applying symmetric extensions to the data, can accurately reconstruct the original data.
                         However, the implementation of the script is a bit problematic. Unless haar wavelets are used, the exact reconstruction is not possible, probably due to the arrangement of the wavelet filter sets in the python package pywt
        :param wavename: The wavelet used to initialize the filter, which for now only supports binary scalar wavelets.
                         Other wavelets or superwavelets, such as a-wavelets, multiwavelets, wavelet frames, curved waves, ridges, strip waves, wavelet frames, etc., are not supported at this time;
                         For 2D/3D data, the corresponding filters are obtained by tensor multiplication of 1D filter banks, and the corresponding wavelets are called tensor wavelets or separable wavelets; to use non-separable wavelets, the reconstruction script
        :param stride: the sampling step, the script must set this value to 2, it is possible to run if it is not set to other values (at this point it is necessary to mask the assert self.stride == 2 in the script), but it does not satisfy wavelet theory;
                        If you are using wavelets of arbitrary binary, such as wavelets of 3 binary, you can adjust this parameter accordingly, but at this time there are more filter sets, which will decompose more high-frequency components accordingly, and the corresponding script content should be updated
        :param in_channels: the number of channels of input data
        :param out_channels: the number of channels of the output data, the default is the same as the number of channels of the input data
        :param groups: the number of groups for the channel dimension, this value needs to be divisible by in_channels.
                        The default value is the same as the number of channels of the input data, i.e. in_channels; the default value here is 1 for general convolution operations
        :param kernel_size: the size of the convolution kernel, this parameter has some conflict with the parameter wavename, that is, the value of this parameter must be greater than the initialized wavelet filter length;
                            The default value of this parameter is equal to the wavelet filter length used for initialization
                            If the filter bank is not updated during training, i.e., the parameter trainable is set to False, it is recommended that the default value of kernel_size be used, because it does not bring any gain except for the increase in the number of operations.
                            If the parameter trainable is set to True, the parameter kernel_size should be greater than or equal to the filter length of the wavelet used for initialization, so that it is possible to train a filter set that is more suitable for the current data distribution.
                            Personally, I don't recommend setting the kernel_size value much larger than the initialized wavelet filter length, I recommend that this value should not exceed 3
        :param trainable: marks whether the filter set parameters are updated during training;
                          If this parameter is set to True and groups is also set to 1, then:
                                DWT layer is equivalent to multiple stride = 2 convolutional layers, but the size of the convolutional kernel and the initialization method are different.
                                The IDWT layer is equivalent to the sum of multiple stride = 2 deconvolution layers, but the size of the convolution kernel and the initialization method are also different.

                When the default values of out_channels and groups are used, the wavelet transform is applied to the input data channel by channel.
                When groups is set to 1, it is similar to the general convolution operation, which can be interpreted as fusing the information in the same frequency band of different channels of the data.
                As with the general convolutional layers, these layers can theoretically handle data of arbitrary size.
                However, if the dimension of the input data is smaller than 1/2 the length of the filter set, an error will be reported when the data is extended during reconstruction.
                In addition, we recommend that the dimensions of the input data be of even values in each dimension.

                Other layers need to be explained in the same way, so we will not explain them again.
        """
        super(DWT_1D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, 'If the filter bank is not updated during training, set kernel_size to the default value None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0, 'The parameter groups should be divisible by in_channels'
        self.stride = stride
        assert self.stride == 2, 'In the current version, stride can only equal 2'
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, 'The value of kernel_size cannot be less than the filter length of the wavelet used for initialization'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low  = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_low = self.filt_low[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        self.filter_high = self.filt_high[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        if torch.cuda.is_available():
            self.filter_low = self.filter_low.cuda()
            self.filter_high = self.filter_high.cuda()
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)

        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 3
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv1d(input, self.filter_low, stride = self.stride, groups = self.groups), \
               F.conv1d(input, self.filter_high, stride = self.stride, groups = self.groups)


class IDWT_1D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        """
            Refer to the description in DWT_1D
            Theoretically, IDWT using simple upsampling and convolution is less computationally intensive and faster than the matrix method.
            However, since simple upsampling is not implemented in Pytorch, the IDWT can only be implemented by deconvolution with [1,0] to achieve simple upsampling.
            This makes the method very much slower than the matrix approach to IDWT.
        """
        super(IDWT_1D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.dec_lo)
        band_high = torch.tensor(wavelet.dec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        up_filter = torch.zeros((self.stride))
        up_filter[0] = 1.0
        up_filter = up_filter[None, None, :].repeat((self.in_channels, 1, 1))
        self.register_buffer('up_filter', up_filter)
        self.filter_low = self.filt_low[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        self.filter_high = self.filt_high[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        if torch.cuda.is_available():
            self.filter_low = self.filter_low.cuda()
            self.filter_high = self.filter_high.cuda()
            self.up_filter = self.up_filter.cuda()
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 0, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1]

    def forward(self, L, H):
        assert len(L.size()) == len(H.size()) == 3
        assert L.size()[0] == H.size()[0]
        assert L.size()[1] == H.size()[1] == self.in_channels
        L = F.pad(F.conv_transpose1d(L, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        H = F.pad(F.conv_transpose1d(H, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        return F.conv1d(L, self.filter_low, stride = 1, groups = self.groups) + \
               F.conv1d(H, self.filter_high, stride = 1, groups = self.groups)


class DWT_2D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        """
            参照 DWT_1D 中的说明
        """
        super(DWT_2D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_ll = self.filt_low[:, None] * self.filt_low[None,:]
        self.filter_lh = self.filt_low[:, None] * self.filt_high[None,:]
        self.filter_hl = self.filt_high[:, None] * self.filt_low[None,:]
        self.filter_hh = self.filt_high[:, None] * self.filt_high[None,:]
        self.filter_ll = self.filter_ll[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_lh = self.filter_lh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hl = self.filter_hl[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hh = self.filter_hh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        if torch.cuda.is_available():
            self.filter_ll = self.filter_ll.cuda()
            self.filter_lh = self.filter_lh.cuda()
            self.filter_hl = self.filter_hl.cuda()
            self.filter_hh = self.filter_hh.cuda()
        if self.trainable:
            self.filter_ll = nn.Parameter(self.filter_ll)
            self.filter_lh = nn.Parameter(self.filter_lh)
            self.filter_hl = nn.Parameter(self.filter_hl)
            self.filter_hh = nn.Parameter(self.filter_hh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv2d(input, self.filter_ll, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_lh, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_hl, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_hh, stride = self.stride, groups = self.groups)


class IDWT_2D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        """
            Refer to the description in DWT_1D
            Theoretically, IDWT using simple upsampling and convolution is less computationally intensive and faster than the matrix method.
            However, since simple upsampling is not implemented in Pytorch, the IDWT can only be implemented by deconvolution with [[1,0],[0,0]] to achieve simple upsampling.
            This makes the method very much slower than the matrix approach to IDWT.
            Currently, in the paper https://arxiv.org/abs/2005.14461, the construction of WaveSNet is in fact still implemented using the matrix method of DWT/IDWT
        """
        super(IDWT_2D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.dec_lo)
        band_high = torch.tensor(wavelet.dec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_ll = self.filt_low[:, None] * self.filt_low[None,:]
        self.filter_lh = self.filt_low[:, None] * self.filt_high[None,:]
        self.filter_hl = self.filt_high[:, None] * self.filt_low[None,:]
        self.filter_hh = self.filt_high[:, None] * self.filt_high[None,:]
        self.filter_ll = self.filter_ll[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_lh = self.filter_lh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hl = self.filter_hl[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hh = self.filter_hh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        up_filter = torch.zeros((self.stride))
        up_filter[0] = 1
        up_filter = up_filter[:, None] * up_filter[None,:]
        up_filter = up_filter[None, None, :, :].repeat(self.out_channels, 1, 1, 1)
        self.register_buffer('up_filter', up_filter)
        if torch.cuda.is_available():
            self.filter_ll = self.filter_ll.cuda()
            self.filter_lh = self.filter_lh.cuda()
            self.filter_hl = self.filter_hl.cuda()
            self.filter_hh = self.filter_hh.cuda()
            #self.up_filter = self.up_filter.cuda()
        if self.trainable:
            self.filter_ll = nn.Parameter(self.filter_ll)
            self.filter_lh = nn.Parameter(self.filter_lh)
            self.filter_hl = nn.Parameter(self.filter_hl)
            self.filter_hh = nn.Parameter(self.filter_hh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1]

    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        assert LL.size()[0] == LH.size()[0] == HL.size()[0] == HH.size()[0]
        assert LL.size()[1] == LH.size()[1] == HL.size()[1] == HH.size()[1] == self.in_channels
        LL = F.conv_transpose2d(LL, self.up_filter, stride = self.stride, groups = self.in_channels)
        LH = F.conv_transpose2d(LH, self.up_filter, stride = self.stride, groups = self.in_channels)
        HL = F.conv_transpose2d(HL, self.up_filter, stride = self.stride, groups = self.in_channels)
        HH = F.conv_transpose2d(HH, self.up_filter, stride = self.stride, groups = self.in_channels)
        LL = F.pad(LL, pad = self.pad_sizes, mode = self.pad_type)
        LH = F.pad(LH, pad = self.pad_sizes, mode = self.pad_type)
        HL = F.pad(HL, pad = self.pad_sizes, mode = self.pad_type)
        HH = F.pad(HH, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv2d(LL, self.filter_ll, stride = 1, groups = self.groups) + \
               F.conv2d(LH, self.filter_lh, stride = 1, groups = self.groups) + \
               F.conv2d(HL, self.filter_hl, stride = 1, groups = self.groups) + \
               F.conv2d(HH, self.filter_hh, stride = 1, groups = self.groups)


class DWT_3D(nn.Module):
    def __init__(self, pad_type = 'replicate', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups=None,
                 kernel_size = None, trainable = False):
        """
            参照 DWT_1D 中的说明
        """
        super(DWT_3D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_lll = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_llh = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_lhl = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_lhh = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_hll = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_hlh = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_hhl = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_hhh = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_lll = self.filter_lll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_llh = self.filter_llh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhl = self.filter_lhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhh = self.filter_lhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hll = self.filter_hll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hlh = self.filter_hlh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhl = self.filter_hhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhh = self.filter_hhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        if torch.cuda.is_available():
            self.filter_lll = nn.Parameter(self.filter_lll).cuda()
            self.filter_llh = nn.Parameter(self.filter_llh).cuda()
            self.filter_lhl = nn.Parameter(self.filter_lhl).cuda()
            self.filter_lhh = nn.Parameter(self.filter_lhh).cuda()
            self.filter_hll = nn.Parameter(self.filter_hll).cuda()
            self.filter_hlh = nn.Parameter(self.filter_hlh).cuda()
            self.filter_hhl = nn.Parameter(self.filter_hhl).cuda()
            self.filter_hhh = nn.Parameter(self.filter_hhh).cuda()
        if self.trainable:
            self.filter_lll = nn.Parameter(self.filter_lll)
            self.filter_llh = nn.Parameter(self.filter_llh)
            self.filter_lhl = nn.Parameter(self.filter_lhl)
            self.filter_lhh = nn.Parameter(self.filter_lhh)
            self.filter_hll = nn.Parameter(self.filter_hll)
            self.filter_hlh = nn.Parameter(self.filter_hlh)
            self.filter_hhl = nn.Parameter(self.filter_hhl)
            self.filter_hhh = nn.Parameter(self.filter_hhh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 5
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv3d(input, self.filter_lll, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_llh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_lhl, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_lhh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hll, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hlh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hhl, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hhh, stride = self.stride, groups = self.groups)


class IDWT_3D(nn.Module):
    def __init__(self, pad_type = 'replicate', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups=None,
                 kernel_size = None, trainable=False):
        """
            Refer to the description in DWT_1D
            Theoretically, IDWT using simple upsampling and convolution is less computationally intensive and faster than the matrix method.
            However, since simple upsampling is not implemented in Pytorch, the IDWT can only be implemented by deconvolution with [[1,0],[0,0]], [[0,0],[0,0]] to achieve simple upsampling.
            This makes the method very much slower than the matrix approach to IDWT.
        """
        super(IDWT_3D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.dec_lo)
        band_high = torch.tensor(wavelet.dec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_lll = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_llh = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_lhl = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_lhh = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_hll = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_hlh = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_hhl = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_hhh = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_lll = self.filter_lll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_llh = self.filter_llh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhl = self.filter_lhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhh = self.filter_lhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hll = self.filter_hll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hlh = self.filter_hlh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhl = self.filter_hhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhh = self.filter_hhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        up_filter = torch.zeros((self.stride))
        up_filter[0] = 1
        up_filter = up_filter[:, None, None] * up_filter[None,:, None] * up_filter[None, None, :]
        up_filter = up_filter[None, None, :, :, :].repeat(self.out_channels, 1, 1, 1, 1)
        self.register_buffer('up_filter', up_filter)
        if torch.cuda.is_available():
            self.filter_lll = nn.Parameter(self.filter_lll).cuda()
            self.filter_llh = nn.Parameter(self.filter_llh).cuda()
            self.filter_lhl = nn.Parameter(self.filter_lhl).cuda()
            self.filter_lhh = nn.Parameter(self.filter_lhh).cuda()
            self.filter_hll = nn.Parameter(self.filter_hll).cuda()
            self.filter_hlh = nn.Parameter(self.filter_hlh).cuda()
            self.filter_hhl = nn.Parameter(self.filter_hhl).cuda()
            self.filter_hhh = nn.Parameter(self.filter_hhh).cuda()
            self.up_filter = self.up_filter.cuda()
        if self.trainable:
            self.filter_lll = nn.Parameter(self.filter_lll)
            self.filter_llh = nn.Parameter(self.filter_llh)
            self.filter_lhl = nn.Parameter(self.filter_lhl)
            self.filter_lhh = nn.Parameter(self.filter_lhh)
            self.filter_hll = nn.Parameter(self.filter_hll)
            self.filter_hlh = nn.Parameter(self.filter_hlh)
            self.filter_hhl = nn.Parameter(self.filter_hhl)
            self.filter_hhh = nn.Parameter(self.filter_hhh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1]

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        assert LLL.size()[0] == LLH.size()[0] == LHL.size()[0] == LHH.size()[0] == HLL.size()[0] == HLH.size()[0] == HHL.size()[0] == HHH.size()[0]
        assert LLL.size()[1] == LLH.size()[1] == LHL.size()[1] == LHH.size()[1] == HLL.size()[1] == HLH.size()[1] == HHL.size()[1] == HHH.size()[1] == self.in_channels
        LLL = F.pad(F.conv_transpose3d(LLL, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LLH = F.pad(F.conv_transpose3d(LLH, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LHL = F.pad(F.conv_transpose3d(LHL, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LHH = F.pad(F.conv_transpose3d(LHH, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HLL = F.pad(F.conv_transpose3d(HLL, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HLH = F.pad(F.conv_transpose3d(HLH, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HHL = F.pad(F.conv_transpose3d(HHL, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HHH = F.pad(F.conv_transpose3d(HHH, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        return F.conv3d(LLL, self.filter_lll, stride = 1, groups = self.groups) + \
               F.conv3d(LLH, self.filter_llh, stride = 1, groups = self.groups) + \
               F.conv3d(LHL, self.filter_lhl, stride = 1, groups = self.groups) + \
               F.conv3d(LHH, self.filter_lhh, stride = 1, groups = self.groups) + \
               F.conv3d(HLL, self.filter_hll, stride = 1, groups = self.groups) + \
               F.conv3d(HLH, self.filter_hlh, stride = 1, groups = self.groups) + \
               F.conv3d(HHL, self.filter_hhl, stride = 1, groups = self.groups) + \
               F.conv3d(HHH, self.filter_hhh, stride = 1, groups = self.groups)


if __name__ == '__main__':
    """
    import numpy as np
    vector = np.array(range(3*2*8)).reshape((3,2,8))
    print(vector)
    wavename = 'haar'
    vector = torch.tensor(vector).float()
    m0 = DWT_1D(wavename = wavename, in_channels = 2, kernel_size = 12, trainable = True)
    L, H = m0(vector)
    print('L size is {}'.format(L.size()))
    print('L is {}'.format(L))
    print('H size is {}'.format(H.size()))
    print('H is {}'.format(H))
    m1 = IDWT_1D(wavename = wavename, in_channels = 2, kernel_size = 12, trainable = True)
    vector_re = m1(L, H)
    print(vector_re)
    print(vector - vector_re)
    print(torch.max(vector - vector_re), torch.min(vector - vector_re))
    """
    """
    import cv2
    import numpy as np
    from DWT_IDWT_layer import DWT_2D as DWT_2D_old
    from DWT_IDWT_layer import IDWT_2D as IDWT_2D_old

    imagename = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(imagename)
    cv2.imshow('image', image)
    image_tensor = torch.tensor(image).float()
    image_tensor.transpose_(dim0 = -1, dim1 = 0).transpose_(dim0 = -1, dim1 = 1)
    image_tensor.unsqueeze_(dim = 0)
    print(image_tensor.size())
    wavename = 'haar'
    m = DWT_2D(wavename = wavename, in_channels = 3, kernel_size = 2, trainable = True)
    m_o = DWT_2D_old(wavename = wavename)
    LL, LH, HL, HH = m(image_tensor)
    LL_o, LH_o, HL_o, HH_o = m_o(image_tensor)
    print('LL == > {}, {}'.format(torch.max(LL - LL_o), torch.min(LL - LL_o)))
    print('LH == > {}, {}'.format(torch.max(LH - LH_o), torch.min(LH - LH_o)))
    print('HL == > {}, {}'.format(torch.max(HL - HL_o), torch.min(HL - HL_o)))
    print('HH == > {}, {}'.format(torch.max(HH - HH_o), torch.min(HH - HH_o)))
    m1 = IDWT_2D(wavename = wavename, in_channels = 3, kernel_size = 2, trainable = True)
    m1_o = IDWT_2D_old(wavename = wavename)
    image_re = m1(LL, LH, HL, HH)
    image_re_o = m1_o(LL_o, LH_o, HL_o, HH_o)
    print('image_re size is {}'.format(image_re.size()))
    print('LL size is {}'.format(LL.size()))
    # print(torch.max(torch.abs(image_tensor - image_re)), torch.min(torch.abs(image_tensor - image_re)))
    image_re.squeeze_().transpose_(dim0 = -1, dim1 = 1).transpose_(dim0 = -1, dim1 = 0)
    image_re_o.squeeze_().transpose_(dim0 = -1, dim1 = 1).transpose_(dim0 = -1, dim1 = 0)
    image_re = np.array(image_re.data)
    image_re_o = np.array(image_re_o)
    cv2.imshow('image_re', image_re / np.max(image_re))

    print(np.max(image - image_re), np.min(image - image_re))
    gap = 0
    gap = gap
    gap_ = -gap if gap != 0 else None
    print(np.max((image - image_re)[gap:gap_, gap:gap_, 2]), np.min((image - image_re)[gap:gap_, gap:gap_, :]))
    cv2.imshow('---', (image - image_re) / np.max(image - image_re))
    cv2.imshow('------', (image - image_re_o) / np.max(image - image_re_o))
    print((image - image_re)[64, 0:20, 0])
    print((image - image_re)[64, -20:, 0])
    print((image_re_o - image_re)[64, 0:20, 0])
    print((image_re_o - image_re)[64, -20:, 0])
    cv2.waitKey(0)
    """
    #"""
    wavename = 'haar'
    vector = torch.ones((1,1,8,8,8))
    m0 = DWT_3D(wavename = wavename, in_channels = 1, kernel_size = 4, trainable = True)
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = m0(vector)
    print('LLL size is {}'.format(LLL.size()))
    m1 = IDWT_3D(wavename = wavename, in_channels = 1, kernel_size = 4, trainable = True)
    vector_re = m1(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
    print(vector - vector_re)
    #"""

