import torch
import torch.nn.functional as F


class ChannelWiseMerger():
    def __init__(self,out_channels,padding,dilation,kernel_size,stride,n_output_crossbars,n_input_crossbars) -> None:
        self.out_channels=out_channels
        self.padding=padding
        self.dilation=dilation
        self.kernel_size=kernel_size
        self.stride=stride
        self.n_output_crossbars=n_output_crossbars
        self.n_input_crossbars=n_input_crossbars
    
    def __call__(self,x, outputs):
        # output shape N,POC,OSpatial_L
        # outputs shape NIC*NOC
        n,partial_oc,ospace=outputs[0].size()
        outputs_cat=torch.cat(outputs,1) # shape N,NIC*NOC POC,OSpatial_L
        outputs_cat=outputs_cat.view(n,self.n_input_crossbars,self.n_output_crossbars*partial_oc,ospace)
        if outputs_cat.size(1)==1:
            output=outputs_cat.reshape(n,-1,ospace)
        else:
            output=outputs_cat.sum(1).reshape(n,-1,ospace) # shape N,ceil(OC),OSpatial_L
        oh=int((x.size(2)+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0]+1)
        output=output.view(n,output.size(1),oh,-1)[:,:self.out_channels]
        return output