import torch
import torch.nn as nn
from quantization.quantizer import ACIQ
from nn_layers.conv import BitwiseStatisticConv2d,CrossbarWiseQuantMappedConv2d

def _fold_bn(conv_module, bn_module):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias

def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b.data)
    else:
        conv_module.bias.data = b.data
    conv_module.weight.data = w.data

def wrap_modules_in_net(net,layer_quantizer,quantizer_kwargs,fuse_bn=False,conv_wrapper=BitwiseStatisticConv2d):
    wrapped_modules={}
    # slice_size=args.active_rows
    
    prev_layer_name=None
    fuse_str='Layer Fuse: '
    for name,m in net.named_modules():
        if isinstance(m,nn.Conv2d):
            _m=conv_wrapper(m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
            _m.quantizer=layer_quantizer(**quantizer_kwargs)
            _m.weight.data=m.weight.data
            _m.bias=m.bias
            _m.mode='raw'
            m.forward_backup=m.forward
            m.forward=_m.forward
            wrapped_modules[name]=_m
        if isinstance(m,nn.BatchNorm2d):
            # print(wrapped_modules)
            if fuse_bn and prev_layer_name in wrapped_modules:
                fuse_str+=f'{name}->{prev_layer_name}; '
                conv=wrapped_modules[prev_layer_name]
                conv.bn_fused=True
                fold_bn_into_conv(conv,m)
                m.forward_back=m.forward
                m.forward=lambda x:x
        prev_layer_name=name
    print(f"{fuse_str}")
    return wrapped_modules

def wrap_modules_to_crossbar(net,rows,cols,layer_quantizer,quantizer_kwargs,fuse_bn=False,conv_wrapper=CrossbarWiseQuantMappedConv2d):
    wrapped_modules={}
    prev_layer_name=None
    fuse_str='Layer Fuse: '
    for name,m in net.named_modules():
        if isinstance(m,nn.Conv2d):
            _m=conv_wrapper(m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,m.bias is not None,m.padding_mode)
            _m.quantizer=layer_quantizer(**quantizer_kwargs)
            _m.weight.data=m.weight.data
            _m.bias=m.bias
            _m.mode='raw'
            m.forward_backup=m.forward
            m.forward=_m.forward
            wrapped_modules[name]=_m
        if isinstance(m,nn.BatchNorm2d):
            # print(wrapped_modules)
            if fuse_bn and prev_layer_name in wrapped_modules:
                fuse_str+=f'{name}->{prev_layer_name}; '
                conv=wrapped_modules[prev_layer_name]
                conv.bn_fused=True
                fold_bn_into_conv(conv,m)
                m.forward_back=m.forward
                m.forward=lambda x:x
        prev_layer_name=name
    for name,m in wrapped_modules.items():
        m.map_to_crossbars(rows,cols)
    print(f"{fuse_str}")
    return wrapped_modules