import torch
from .input_selector import ChannelWiseSelector
from .output_merger import ChannelWiseMerger
from PIMmodel.crossbar import BaseCrossbar
import torch.nn as nn

class BaseConvMapper():
    def __init__(self,rows,cols,n_cell_per_weight=1) -> None:
        self.rows=rows
        self.cols=cols
    
    def __call__(self,conv:nn.Conv2d):
        weight=conv.weight.data
        assert conv.groups==1
        oc,ic,kh,kw=weight.size()
        ic_per_crossbar=self.rows//(kw*kh)
        n_input_crossbars=0
        
        crossbars=[]
        
        for ic_st in range(0,ic,ic_per_crossbar):
            ic_ed=min(ic_st+ic_per_crossbar,ic)
            n_input_crossbars+=1
            for oc_st in range(0,oc,self.cols):
                oc_ed=min(oc_st+self.cols,oc)
                
                
                _w=weight[oc_st:oc_ed,ic_st:ic_ed].view(oc_ed-oc_st,-1) # shape OC,IC×∏(kernel_size)
                #debug
                #print(_w)
                
                w_local=torch.zeros([self.cols,self.rows]).to(weight.device)
                w_local[:_w.size(0),:_w.size(1)]=_w
                #debug
                #print(w_local)
                
                selector=ChannelWiseSelector(ic_st,ic_ed,conv.kernel_size,conv.dilation,conv.padding,conv.stride)
                crossbar=BaseCrossbar(w_local,input_selector=selector)
                crossbars.append(crossbar)
        n_output_crossbars=len(crossbars)//n_input_crossbars
        #print(f"Map {conv} to {len(crossbars)} crossbars (R={self.rows} C={self.cols})")
        merger=ChannelWiseMerger(conv.out_channels,conv.padding,conv.dilation,conv.kernel_size,conv.stride,n_output_crossbars,n_input_crossbars)
        return crossbars,merger