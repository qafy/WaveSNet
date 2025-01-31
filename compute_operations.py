import torch

count_ops = 0

def measure_layer(layer, x, multi_add=1):
    type_name = str(layer)[:str(layer).find('(')].strip()
    if type_name in ['Conv2d']:
        print(type_name, '==>', layer.stride[0] + 1)
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
        delta_ops = x.size()[1] * out_w * out_h * kernel_ops

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.numel()

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = weight_ops + bias_ops

    elif type_name in ['BatchNorm2d']:
        normalize_ops = x.numel()
        scale_shift = normalize_ops
        delta_ops = normalize_ops + scale_shift

    ### ops_nothing
    elif type_name in ['Dropout2d', 'DropChannel', 'Dropout']:
        delta_ops = 0

    ### unknown layer type
    else:
        delta_ops = 0
        print('unknown layer type: %s' % type_name)

    global count_ops
    count_ops += delta_ops
    return

def is_leaf(module):
    return sum(1 for x in module.children()) == 0

# Determine if the module is a node that needs to calculate flops
def should_measure(module):
    # The residual structure in the code may define a Sequential with empty content
    if str(module).startswith('Sequential'):
        return False
    if is_leaf(module):
        return True
    return False

def measure_model(model, shape=(1,3,224,224)):
    global count_ops
    data = torch.zeros(shape)

    # Integrate the operation of calculating flops into the forward function
    def new_forward(m):
        def lambda_forward(x):
            measure_layer(m, x)
            return m.old_forward(x)
        return lambda_forward

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                # Add an old_forward property to save the default forward function
                # Facilitate the recovery of the forward function after the flops are calculated
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # Recovery of the modified forward function
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    # The global variable count_ops is updated during forward
    model.forward(data)
    restore_forward(model)

    return count_ops

# IMPORTANT This file is for measuring FLOP performance of the specific models
# Not needed for qualitative results
from SegNet_UNet.U_Net import unet_vgg16
from SegNet_UNet.SegNet import segnet_vgg16
if __name__ == '__main__':
    net = unet_vgg16()
    print(measure_model(net, shape = (2,3,768,768)))

    #resnet_dwt_ma()
    #vgg_dwt_ma()
    #dense_dwt_ma()
    #for no in ResNet_Nos:
    #    print(221727744 / (221727744+no))
    #print(1430542848 / (1430542848+15510949352))
    #print(177999360 / (177999360+2881817320))
