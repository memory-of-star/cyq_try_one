import torch
from .quantizer import EasyQuant


class DFQ(EasyQuant):
    """
    """
    def __init__(self, w_bit, a_bit,channel_wise=False,eq_alpha=0.5,eq_beta=2,eq_n=100) -> None:
        super().__init__(w_bit, a_bit, channel_wise=channel_wise, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n)
        self.next_layer=None
        self.now_layer=None

    def weight_equalization(self,weight1,bias1,weight2,bias2):
        out1_c=weight1.size(0)
        r1=weight1.view(out1_c,-1).abs().max(1)[0]
        r2=weight2.view(weight2.size(0),out1_c,-1).transpose(0,1).reshape(out1_c,-1).max(1)[0]
        s=torch.sqrt(r1*r2)/r2
        new_w1=weight1/s.view(-1,*[1]*(weight1.dim()-1))
        new_w2=weight2.view(weight2.size(0),out1_c,-1)*s.view(1,-1,1)
        new_w2=new_w2.view(weight2.size())
        if bias1 is not None:
            new_b1=bias1/s
        else:
            new_b1=None
        new_b2=bias2
        return new_w1,new_b1,new_w2,new_b2

    def calibration(self,x,weight,bias,op):
        # step1: collection the FP32 values
        if self.calibration_step==1:
            if self.next_layer is None or self.now_layer is None:
                print("Debug: Not Weight Equalize layer")
                pass
            else:
                weight2,bias2=self.next_layer.weight,self.next_layer.bias
                weight,bias,w2,b2=self.weight_equalization(weight,bias,weight2,bias2)
                self.next_layer.weight[...]=w2
                if b2 is not None:
                    self.next_layer.bias[...]=b2
                self.now_layer.weight[...]=weight
            if bias is not None:
                self.now_layer.bias[...]=bias
            out=op(x,weight,bias)
            self.raw_outs.append(out.cpu().detach())
            return out
        # step1: search for the best S^w of each layer
        elif self.calibration_step==2:
            # initialize
            if self.channel_wise:
                max=weight.data.abs().view(weight.size(0),-1).max(1)[0]
                max=max.view(-1,*[1]*(weight.dim()-1))
            else:
                max=weight.data.abs().max()
            init_interval=max/(2**(self.w_bit-1)-0.5) # symmetric quantization
            raw_out=torch.cat(self.raw_outs,0).to(x.device)
            self.weight_interval,best_out=self.search_best_weight(x,weight,bias,op,raw_out,init_interval)
            print(f"Set weight_interval={self.weight_interval.view(-1)}")
            return best_out
        # step3: search for the best S^a of each layer
        elif self.calibration_step==3:
            w_sim,b_sim=self.quant_weight_bias(weight,bias)
            # initialize
            if self.channel_wise:
                max=x.data.abs().transpose(0,1).reshape(x.size(1),-1).max(1)[0]
                max=max.view(1,-1,*[1]*(weight.dim()-2))
            else:
                max=x.data.abs().max()
            init_interval=max/(2**(self.a_bit-1)-0.5) # symmetric quantization
            raw_out=torch.cat(self.raw_outs,0).to(x.device)
            self.input_interval,best_out=self.search_best_input(x,w_sim,b_sim,op,raw_out,init_interval)
            print(f"Set input_interval={self.input_interval.view(-1)}")
            return best_out
    
class PowerOf2DFQ(DFQ):
    """
    """
    def __init__(self, w_bit, a_bit,channel_wise=False,eq_alpha=0.5,eq_beta=2,eq_n=100) -> None:
        super().__init__(w_bit, a_bit, channel_wise=channel_wise, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n)
        self.next_layer=None
        self.now_layer=None