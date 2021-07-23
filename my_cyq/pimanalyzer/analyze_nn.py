import torch
import torch.nn as nn
import sys
import torchvision.models.resnet as resnet
import datasets
import argparse
import nn_layers.conv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import argparse
from quantization import quantizer
import net_wrap
from quantization.quantizer import quant_calib

def load_net(name):
    if name == 'resnet18':
        net=resnet.resnet18(pretrained=True)
    else:
        raise NotImplementedError
    net=net.cuda()
    return net

def load_datasets(name,data_root,calib_size=128):
    if name=='imagenet':
        g=datasets.ImageNetLoaderGenerator(data_root,'imagenet',calib_size,128,4)
        test_loader=g.test_loader(shuffle=True,batch_size=16)
        calib_loader=g.calib_loader(calib_size)
    elif name=='cifar10':
        g=datasets.CIFARLoaderGenerator(data_root,'cifar10',calib_size,128,4)
        test_loader=g.test_loader(shuffle=False)
        calib_loader=g.calib_loader(calib_size)
    else:
        raise NotImplementedError
    return test_loader,calib_loader

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('net_name',type=str)
    parser.add_argument('dataset',type=str)
    parser.add_argument('--data_root',default="data",type=str)
    parser.add_argument('--quantizer',default="aciq",type=str)
    parser.add_argument('--statistic_num',type=int,default=32)
    parser.add_argument('--out_path',type=str,default='output')
    # PIM-based settings
    parser.add_argument('--active_rows',default=8,type=int)
    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    net=load_net(args.net_name)
    test_loader,calib_loader=load_datasets(args.dataset,args.data_root)
    if args.quantizer=='aciq':
        layer_quantizer=quantizer.ACIQ
    elif args.quantizer=='easyquant':
        layer_quantizer=quantizer.EasyQuant
    elif args.quantizer=='dfq':
        layer_quantizer=quantizer.DFQ
    wrapped_modules=net_wrap.wrap_modules_in_net(net,layer_quantizer=layer_quantizer)
    quant_calib(net,wrapped_modules,calib_loader)

    for name,module in wrapped_modules.items():
        module.mode='statistic_forward'
    cnt=0
    with torch.no_grad():
        for inp,target in test_loader:
            inp=inp.cuda()
            net(inp)
            cnt+=inp.size(0)
            if cnt>=args.statistic_num:
                break
    
    zero_out_exclude_in_zero_tot=0
    tot_out_exclude_in_zero=0

    tot_zero_in=0
    tot_in=0

    tot_zero_out=0
    tot_out=0
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    for name,module in wrapped_modules.items():
        # print(module.statistic)
        s=module.statistic
        
        torch.save(s,f'{args.out_path}/{name}.pth')
        
        in_zero_frac=[s[f'zero_in_{i}']/s['in_num'] for i in range(module.act_bits)]
        print(f"{name} in_zero/tot {in_zero_frac}")
        tot_zero_in+=np.sum([s[f'zero_in_{i}'] for i in range(module.act_bits)])
        tot_in+=np.sum([s['in_num'] for i in range(module.act_bits)])
        

        out_zero_frac=[s[f'zero_out_{i}']/s['out_num'] for i in range(module.weight_bits)]
        print(f"{name} out_zero/tot {out_zero_frac}")
        tot_zero_out+=np.sum([s[f'zero_out_{i}'] for i in range(module.act_bits)])
        tot_out+=np.sum([s[f'out_num'] for i in range(module.act_bits)])

        out_zero_frac_exclude_in_zero=[s[f'zero_out_{i}_exclude_in_zero']/(s[f'tot_out_{i}_exclude_in_zero']) for i in range(module.weight_bits)]
        print(f"{name} out_zero_exclude_in_zero/tot_exclude_in_zero {out_zero_frac_exclude_in_zero}")
        zero_out_exclude_in_zero_tot+=np.sum([s[f'zero_out_{i}_exclude_in_zero']/1 for i in range(module.weight_bits)])
        tot_out_exclude_in_zero+=np.sum([s[f'tot_out_{i}_exclude_in_zero'] for i in range(module.weight_bits)])

    print("zero_out_exclude_in_zero_tot/tot_out_exclude_in_zero",zero_out_exclude_in_zero_tot/tot_out_exclude_in_zero)
    print("tot_zero_in/tot_in",tot_zero_in/tot_in)
    print("tot_zero_out/tot_out",tot_zero_out/tot_out)

