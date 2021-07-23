import torch.nn.functional as F


class ChannelWiseSelector():
    def __init__(self,c_st,c_ed,k_size,dilation,padding,stride) -> None:
        self.c_st=c_st
        self.c_ed=c_ed
        self.k_size=k_size
        self.dilation=dilation
        self.padding=padding
        self.stride=stride
    
    def __call__(self, x):
        x_c=x[:,self.c_st:self.c_ed]
        x_unfolded=F.unfold(x_c,self.k_size,self.dilation,self.padding,self.stride) # shape N,IC×∏(kernel_size),OSpatial_L
        return x_unfolded
        