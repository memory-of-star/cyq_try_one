
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization.quant_functions import *
from mapper.conv_mapper import BaseConvMapper
import numpy as np
from tqdm import tqdm
from quantization.sublayer_quantizer import SubLayerQuantizer
import random
import copy


class QuantizeConv2d(nn.Conv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        
        self.mode=None
        self.quantizer=None
        self.bn_fused=False
    
    def forward(self, x):
        if self.mode=='raw':
            out=super().forward(x)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        elif self.mode=="calibration_forward":
            out=self.calibrate_forward(x)
        elif self.mode=='statistic_forward':
            out=self.statistic_forward(x)
        elif self.mode=='mapped_forward':
            out=self.mapped_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self,x):
        assert self.quantizer.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        weight_sim,bias_sim=self.quantizer.quant_weight_bias(self.weight,self.bias)
        x_sim=self.quantizer.quant_activation(x)
        out_sim=F.conv2d(x_sim, weight_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        out_sim=self.quantizer.quant_output(out_sim)
        return out_sim
    
    def get_quant_weight_bias(self):
        return self.quantizer.quant_weight_bias(self.weight,self.bias)

    def calibrate_forward(self,x):
        # assert self.weight_bits is not None and self.act_bits is not None, f"You should set the weight_bits and bias_bits for {self}"
        op=lambda input,weight,bias:F.conv2d(input,weight,bias,self.stride,self.padding, self.dilation, self.groups)
        out_sim=self.quantizer.calibration(x,self.weight,self.bias,op)
        return out_sim

    def mapped_forward(self,x):
        assert self.crossbars is not None,f"You should map the conv to the crossbar before using mapped_forward"
        outputs=[]
        for crossbar in self.crossbars:
            output=crossbar(x)
            outputs.append(output)
        rst=self.merger(x,outputs)
        # bias
        if self.bias:
            rst+=self.bias.view(1,-1,1,1)
        return rst

class BitwiseStatisticConv2d(QuantizeConv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.slice_size=None
        self.statistic={}
        
    def statistic_forward(self,x):
        print(f"Run statistic_forward of {self} with input {x.size()}")
        weight_sim,bias_sim=self.quantizer.quant_weight_bias(self.weight,self.bias)
        w_integer=weight_sim.integer
        x_sim=self.quantizer.quant_activation(x)
        in_integer=x_sim.integer
        out_sim=F.conv2d(x_sim, weight_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        out_sim=self.quantizer.quant_output(out_sim)
        
        
        # if 'in_quant' in self.activation_quant_mode:
        #     if self.activation_quant_mode=='in_quant_unsigned':
        #         in_max_int=2**(self.act_bits)-1
        #     else:
        #         in_max_int=2**(self.act_bits-1)-1
        #     in_integer=torch.round_(x/self.x_scale).clamp_(-in_max_int,in_max_int)
        #     x=in_integer*self.x_scale
        # w_max_int=2**(self.weight_bits)-1
        # w_integer=torch.round_(self.weight.data/self.weight_scale).clamp_(-w_max_int,w_max_int)
        # w_q=w_integer*self.weight_scale
        # raw_out=F.conv2d(x,w_q,self.bias,self.stride,self.padding,self.dilation,self.groups)
        
        # b,oc,oh,ow=raw_out.size()
        b,oc,oh,ow=out_sim.size()
        
        kernel_size=self.weight.size()[2:]
        x_unfolded=F.unfold(in_integer,kernel_size,self.dilation,self.padding,self.stride) # shape N,C×∏(kernel_size),L
        W=w_integer.view(oc,-1) # shape oc,C*∏(kernel_size)
        b,win_size,n_window=x_unfolded.size()
        n_slice=win_size//self.slice_size
        # ignore the un-aligned datas
        x_unfolded=x_unfolded[:,:n_slice*self.slice_size]
        W=W[:,:n_slice*self.slice_size]
        
        all_x_slice=x_unfolded.view(b,n_slice,self.slice_size,n_window)
        
        W_slice=W.view(oc,n_slice,self.slice_size)
        # print(f"x_slice {x_slice.size()} W_slice {W_slice.size()}")
        S=0 # shape N,oc,L
        all_x_slice=all_x_slice.long()
        W_slice=W_slice.long()
        if f'in_num' not in self.statistic:
            self.statistic[f'in_num']=0
        self.statistic['in_num']+=b*n_slice*n_window
        if f'out_num' not in self.statistic:
            self.statistic[f'out_num']=0
        self.statistic['out_num']+=b*n_slice*oc*n_window*self.act_bits
        with torch.no_grad():
            for act_bit_i in range(self.act_bits):
                x_bit=((all_x_slice>>act_bit_i)&1).float()
                zero_in_num=torch.sum((torch.sum(x_bit,2)==0).long()).item()
                if f'zero_in_{act_bit_i}' not in self.statistic:
                    self.statistic[f'zero_in_{act_bit_i}']=0
                self.statistic[f'zero_in_{act_bit_i}']+=zero_in_num
                
                for w_bit_i in range(self.weight_bits):
                    w_bit=((W_slice>>w_bit_i)&1).float()
                    zero_out_num=0
                    for i in range(n_slice):
                        psum=torch.matmul(w_bit[:,i],x_bit[:,i])
                        # if i not in self.statistic:
                        #     self.statistic[i]=[]
                        # psum_sorted=torch.sort(psum.view(-1))[0][::int(psum.view(-1).size(0)/1000)]
                        # self.statistic[i].append(psum_sorted.detach().cpu().numpy())
                        zero_out_num+=torch.sum((psum==0).long()).item()
                    if f'zero_out_{w_bit_i}' not in self.statistic:
                        self.statistic[f'zero_out_{w_bit_i}']=0
                    self.statistic[f'zero_out_{w_bit_i}']+=zero_out_num
                    if f'zero_out_{w_bit_i}_exclude_in_zero' not in self.statistic:
                        self.statistic[f'zero_out_{w_bit_i}_exclude_in_zero']=0
                    # shape of zero out: n_slice*(b*oc*L); shape of zero in: b*n_slice*L
                    self.statistic[f'zero_out_{w_bit_i}_exclude_in_zero']+=zero_out_num-oc*zero_in_num
                    if f'tot_out_{w_bit_i}_exclude_in_zero' not in self.statistic:
                        self.statistic[f'tot_out_{w_bit_i}_exclude_in_zero']=0
                    # shape of zero out: n_slice*(b*oc*L); shape of zero in: b*n_slice*L
                    self.statistic[f'tot_out_{w_bit_i}_exclude_in_zero']+=psum.numel()*n_slice-oc*zero_in_num
                
        return out_sim

class BaseMappedConv2d(nn.Conv2d):
    """
    Map the nn.Conv2d to different size of crossbar
    """
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        
        self.mode='raw'
        # self.crossbar_cols=None
        # self.crossbar_rows=None
        # self.n_cell_per_weight=None
        # self.n_input_steps=None
        self.crossbars=None
        self.merger=None
    
    def map_to_crossbars(self,rows,cols,n_cell_per_weight=1,n_input_steps=1):
        mapper=BaseConvMapper(rows,cols,n_cell_per_weight)
        self.crossbars,self.merger=mapper(self)
        self.rows=rows
        self.cols=cols

    def forward(self, x):
        if self.mode=='raw':
            out=super().forward(x)
        elif self.mode=="mapped_forward":
            out=self.mapped_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def mapped_forward(self,x):
        assert self.crossbars is not None,f"You should map the conv to the crossbar before using mapped_forward"
        outputs=[]
        for crossbar in self.crossbars:
            output=crossbar(x)
            outputs.append(output)
        rst=self.merger(x,outputs)
        # bias
        if self.bias:
            rst+=self.bias.view(1,-1,1,1)
        return rst

class LayerWiseQuantMappedConv2d(QuantizeConv2d,BaseMappedConv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
    
    def quant_forward(self,x):
        assert self.crossbars is not None
        assert self.quantizer.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        
        x_sim=self.quantizer.quant_activation(x)
        out_sims=[]
        for crossbar in self.crossbars:
            weight_sim,_=self.quantizer.quant_weight_bias(crossbar.weight,None)
            out_sim=crossbar(x_sim,weight_sim)
            out_sims.append(out_sim)
        out_sim=self.merger(x,out_sims)
        out_sim=self.quantizer.quant_output(out_sim)
        if self.bias is not None:
            out_sim+=self.bias.view(1,-1,1,1)
        return out_sim
    
    def calibrate_forward(self,x):
        assert self.weight_bits is not None and self.act_bits is not None, f"You should set the weight_bits and bias_bits for {self}"
        out_sims=[]
        bias=None
        weights=[]
        def op(input,weights,bias):
            for i,crossbar in enumerate(self.crossbars):
                out_sim=crossbar(input,weights[i*self.cols:(i+1)*self.cols])
                out_sims.append(out_sim)
            out_sim=self.merger(x,out_sims)
            return out_sim
        for crossbar in self.crossbars:
            weights.append(crossbar.weight)
        weights=torch.cat(weights,0)
        out_sim=self.quantizer.calibration(x,weights,bias,op)
        if self.bias is not None:
            out_sim+=self.bias.view(1,-1,1,1)
        return out_sim

class CrossbarWiseQuantMappedConv2d(LayerWiseQuantMappedConv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
    
    def quant_forward(self,x):
        assert self.crossbars is not None
        assert self.quantizer.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        out_sims=[]
        for i,crossbar in enumerate(self.crossbars):
            weight_sim,_=self.quantizer[i].quant_weight_bias(crossbar.weight,None)
            x_i=self.quantizer[i].quant_activation(x)
            out_sim=crossbar(x_i,weight_sim)
            out_sim=self.quantizer[i].quant_output(out_sim)
            out_sims.append(out_sim)
        out_sim=self.merger(x,out_sims)
        if self.bias is not None:
            out_sim+=self.bias.view(1,-1,1,1)
        return out_sim
    
    def calibrate_forward(self,x):
        if not isinstance(self.quantizer,SubLayerQuantizer):
            self.quantizer=SubLayerQuantizer(len(self.crossbars),self.quantizer)
        out_sims=[]
        bias=None
        for i,crossbar in enumerate(self.crossbars):
            op=lambda input,weight,bias: crossbar(input,weight)
            out_sim=self.quantizer[i].calibration(x,crossbar.weight.data,bias,op)
            # print(f"CF for {i} self.quantizer[i]={self.quantizer[i].weight_interval}")
            out_sims.append(out_sim)
        out_sim=self.merger(x,out_sims)
        if self.bias is not None:
            out_sim+=self.bias.view(1,-1,1,1)
        return out_sim


class ReorderingCrossbarWiseQuantMappedConv2d(CrossbarWiseQuantMappedConv2d):
    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.now_layer = self
        self.next_layer = None
        self.best_similarity = -1e9
        self.best_reordering_weight = None
        self.next_layer_best_reordering_weight = None
        self.outputfile = None

    def calibrate_forward(self,x):
        if not isinstance(self.quantizer,SubLayerQuantizer):
            self.quantizer=SubLayerQuantizer(len(self.crossbars),self.quantizer)
        out_sims=[]
        bias=None

        for i,crossbar in enumerate(self.crossbars):
            op=lambda input,weight,bias: crossbar(input,weight)
            out_sim=self.quantizer[i].calibration(x,crossbar.weight.data,bias,op)
            # print(f"CF for {i} self.quantizer[i]={self.quantizer[i].weight_interval}")
            out_sims.append(out_sim)
        out_sim=self.merger(x,out_sims)
        if self.bias is not None:
            out_sim+=self.bias.view(1,-1,1,1)

        #when quantizing weight
        print(self.quantizer[0].calibration_step)
        print(self.next_layer == None)
        if self.quantizer[0].calibration_step==2 and (not self.next_layer == None):
            self.mode = 'raw'
            out = self.forward(x)
            self.mode = 'calibration_forward'
            similarity=F.cosine_similarity(out.reshape(-1), out_sim.reshape(-1), 0)
            print("similarity:", similarity)
            print("best_similarity:", self.best_similarity)
            if self.outputfile != None:
                self.outputfile.write("similarity:", similarity, '\n')
                self.outputfile.write("best_similarity:", self.best_similarity, '\n')

            if similarity > self.best_similarity:
                #TODO:to speedup, I shall reserve it then
                #self.best_reordering_weight = copy.deepcopy(self.weight)
                #self.next_layer_best_reordering_weight = copy.deepcopy(self.next_layer.weight)
                self.best_similarity = similarity
        
        return out_sim

    def change_row(self, a, b):
        self.weight.data[[a, b]] = self.weight.data[[b, a]].contiguous()
        self.next_layer.weight.data = self.next_layer.weight.data.transpose(0, 1)
        self.next_layer.weight.data[[a, b]] = self.next_layer.weight.data[[b, a]].contiguous()
        self.next_layer.weight.data = self.next_layer.weight.data.transpose(0, 1).contiguous()

    def shuffle_row(self, times, channels):
        for i in range(times):
            a = random.randint(0, channels-1)
            b = random.randint(0, channels-1)
            if not (a == b or self.next_layer == None):
                self.change_row(a, b) 
            

    def reorder(self, command, channels, random_times=30, your_next_layer=None):
        if self.next_layer == None and (not your_next_layer == None):
            self.next_layer = your_next_layer
        if command == 'random_reorder':
            self.shuffle_row(random_times, channels)
        if command == 'mc_reorder':
            pass
        self.remap()

    def remap(self):
        self.map_to_crossbars(self.rows, self.cols)
        if isinstance(self.next_layer, ReorderingCrossbarWiseQuantMappedConv2d):
            self.next_layer.map_to_crossbars(self.rows, self.cols)

    
