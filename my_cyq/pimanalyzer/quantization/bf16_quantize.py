import torch.nn as nn
import torch


def bf16_quantize_model(quant_mode,model):
    quantize_layers=[]
    if quant_mode=='bf16':
        def bf16_hook(m,i,o):
            # m.weight.data=m.weight.data.bfloat16().float()
            i=i[0].bfloat16().float()
            o=m.forward(i)
            return o
        
        for name,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                m.weight.data=m.weight.data.bfloat16().float()
                m.bias.data=m.bias.data.bfloat16().float()
                quantize_layers+=[name]
                m._forward_hooks.clear()
                m.register_forward_hook(bf16_hook)
    elif quant_mode=='layerwise_bf16':
        def bf16_layerwise_hook(m,i,o):
            # m.weight.data=m.weight.data.bfloat16().float()
            i=i[0]
            a_max=torch.amax(i.abs(),[1,2,3],keepdim=True)
            # a_max=i.abs().max()
            exponent=torch.ceil(torch.log2(a_max))
            interval=2**exponent/128
            new_a=torch.round(i/interval)*interval
            o=m.forward(new_a)
            return o

        for name,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                w_max=m.weight.data.abs().max()
                exponent=torch.ceil(torch.log2(w_max))
                interval=2**exponent/128
                new_weight=torch.round(m.weight.data/interval)*interval
                m.weight.data=new_weight
                m.bias.data=m.bias.data.bfloat16().float()
                # print(f"transform weight {name} exponent {exponent}")
                quantize_layers+=[name]
                m._forward_hooks.clear()
                m.register_forward_hook(bf16_layerwise_hook)
    elif quant_mode=='channelwise_bf16':
        def bf16_channelwise_hook(m,i,o):
            # m.weight.data=m.weight.data.bfloat16().float()
            i=i[0]
            a_max=torch.amax(i.abs(),[2,3],keepdim=True)+1e-9
            # a_max=torch.amax(i.abs(),[1,2,3],keepdim=True)
            # a_max=torch.amax(i.abs(),[0,2,3],keepdim=True)
            # a_max=torch.max(i.abs())
            # print(a_max.size())
            # a_max=i.abs().max([0,2,3],keepdim=True)
            exponent=torch.ceil(torch.log2(a_max))
            interval=2**exponent/128
            new_a=torch.round(i/interval)*interval
            o=m.forward(new_a)
            return o

        for name,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                w_max=torch.amax(m.weight.data.abs(),[2,3],keepdim=True)+1e-9
                # w_max=m.weight.data.abs().max([0,2,3],keepdim=True)
                exponent=torch.ceil(torch.log2(w_max))
                interval=2**exponent/128
                new_weight=torch.round(m.weight.data/interval)*interval
                m.weight.data=new_weight
                m.bias.data=m.bias.data.bfloat16().float()
                # print(f"transform weight {name} exponent {exponent.flatten()}")
                quantize_layers+=[name]
                m._forward_hooks.clear()
                m.register_forward_hook(bf16_channelwise_hook)
    elif quant_mode=='channelwise_weightonly_bf16':
        def bf16_channelwise_weight_only_hook(m,i,o):
            # m.weight.data=m.weight.data.bfloat16().float()
            i=i[0]
            # a_max=torch.amax(i.abs(),[2,3],keepdim=True)
            # a_max=torch.amax(i.abs(),[1,2,3],keepdim=True)
            a_max=torch.amax(i.abs(),[1,2,3],keepdim=True)
            # print(a_max.size())
            # a_max=i.abs().max([0,2,3],keepdim=True)
            exponent=torch.ceil(torch.log2(a_max))
            interval=2**exponent/128
            new_a=torch.round(i/interval)*interval
            o=m.forward(new_a)
            return o

        for name,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                w_max=torch.amax(m.weight.data.abs(),[1,2,3],keepdim=True)
                # w_max=m.weight.data.abs().max([0,2,3],keepdim=True)
                exponent=torch.ceil(torch.log2(w_max))
                interval=2**exponent/128
                new_weight=torch.round(m.weight.data/interval)*interval
                m.weight.data=new_weight
                m.bias.data=m.bias.data.bfloat16().float()
                # print(f"transform weight {name} exponent {exponent.flatten()}")
                quantize_layers+=[name]
                m._forward_hooks.clear()
                m.register_forward_hook(bf16_channelwise_weight_only_hook)
    else:
        raise NotImplementedError()
    print(f"quantize_layers {quantize_layers}")