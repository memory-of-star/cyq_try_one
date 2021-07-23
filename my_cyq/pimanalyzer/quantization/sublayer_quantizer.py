from .quantizer import BaseQuantizer
import copy

class SubLayerQuantizer():
    def __init__(self,n_quantizer,template_quantizer) -> None:
        self.quantizers=[]
        for i in range(n_quantizer):
            self.quantizers.append(copy.deepcopy(template_quantizer))
    
    @property 
    def n_calibration_steps(self):
        return max([_.n_calibration_steps for _ in self.quantizers])

    @property
    def calibrated(self):
        return min([_.calibrated for _ in self.quantizers])
    
    @property
    def calibration_step(self):
        return max([_.calibration_step for _ in self.quantizers])

    @calibration_step.setter
    def calibration_step(self,value):
        for q in self.quantizers:
            q.calibration_step=value
    
        
    def __getitem__(self,i):
        return self.quantizers[i]

    def clear_raw_outs(self):
        for q in self.quantizers:
            q.raw_outs.clear()

    def clear_raw_inputs(self):
        for q in self.quantizers:
            q.raw_inputs.clear()