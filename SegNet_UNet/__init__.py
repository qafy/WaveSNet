__all__ = ['My_Sequential', 'My_Sequential_re', 'My_Sequential_3D', 'My_Sequential_re_3D']

from collections import OrderedDict
from itertools import islice
import operator
from torch.nn import Module
from torch import nn
from DWT_IDWT.DWT_IDWT_layer import *
from SamplingOperations.sampling import *
import torch.nn.functional as F


class My_Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    若某个模块输出多个数据，只将第一个数据往下传
    """

    def __init__(self, *args):
        super(My_Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(My_Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        self.output = []
        for module in self._modules.values():
            input = module(input)
            if isinstance(input, tuple):
                assert len(input) == 4 or len(input) == 2 or len(input) == 5
                self.output.append(input[1:])
                input = input[0]
        if self.output != []:
            return input, self.output
        else:
            return input


class My_Sequential_re(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    若某个模块输出多个数据，只将第一个数据往下传
    """

    def __init__(self, *args):
        super(My_Sequential_re, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        self.output = []

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(My_Sequential_re, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, *input):
        LL = input[0]
        index = 1
        for module in self._modules.values():
            if isinstance(module, My_UpSampling_IDWT):
                LH = input[index]
                HL = input[index + 1]
                HH = input[index + 2]
                feature_map = input[index + 3]
                LL = module(LL, LH, HL, HH, feature_map = feature_map)
                index += 4
            elif isinstance(module, IDWT_2D) or 'idwt' in dir(module):
                LH = input[index]
                HL = input[index + 1]
                HH = input[index + 2]
                LL = module(LL, LH, HL, HH)
                index += 3
            elif isinstance(module, nn.MaxUnpool2d):
                indices = input[index]
                LL = module(input = LL, indices = indices)
                #_, _, h, w = LL.size()
                #LL = F.interpolate(LL, size = (2*h, 2*w), mode = 'bilinear', align_corners = True)
                index += 1
            elif isinstance(module, My_UpSampling_SC):
                feature_map = input[index]
                LL = module(input = LL, feature_map = feature_map)
                index += 1
            else:
                LL = module(LL)
        return LL

# extended 

class My_Sequential_3D(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    若某个模块输出多个数据，只将第一个数据往下传
    """

    def __init__(self, *args):
        super(My_Sequential_3D, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(My_Sequential_3D, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        self.output = []
        for module in self._modules.values():
            input = module(input)
            if isinstance(input, tuple):
                assert len(input) == 4 or len(input) == 2 or len(input) == 5
                self.output.append(input[1:])
                input = input[0]
        if self.output != []:
            return input, self.output
        else:
            return input
        
class My_Sequential_re_3D(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    若某个模块输出多个数据，只将第一个数据往下传
    """

    def __init__(self, *args):
        super(My_Sequential_re_3D, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        self.output = []

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(My_Sequential_re_3D, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, *input):
        LLL = input[0]
        index = 1
        for module in self._modules.values():
            if isinstance(module, My_UpSampling_IDWT_3D):
                LLH = input[index]
                LHL = input[index + 1]
                LHH = input[index + 2]
                HLL = input[index + 3]
                HLH = input[index + 4]
                HHL = input[index + 5]
                HHH = input[index + 6]
                feature_map = input[index + 7]
                LLL = module(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH, feature_map = feature_map)
                index += 8
            elif isinstance(module, IDWT_3D) or 'idwt' in dir(module):
                LLH = input[index]
                LHL = input[index + 1]
                LHH = input[index + 2]
                HLL = input[index + 3]
                HLH = input[index + 4]
                HHL = input[index + 5]
                HHH = input[index + 6]
                LLL = module(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
                index += 7
            elif isinstance(module, nn.MaxUnpool3d):
                indices = input[index]
                LLL = module(input = LLL, indices = indices)
                #_, _, h, w = LL.size()
                #LL = F.interpolate(LL, size = (2*h, 2*w), mode = 'bilinear', align_corners = True)
                index += 1
            elif isinstance(module, My_UpSampling_SC_3D):
                feature_map = input[index]
                LLL = module(input = LLL, feature_map = feature_map)
                index += 1
            else:
                LLL = module(LLL)
        return LLL