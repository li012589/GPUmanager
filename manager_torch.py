from  manager import GPUmanager

import torch

class torchGPUmanager(GPUmanager):
    def __init__(self,qargs=[]):
        super(tfGPUmanager,self).__init__(qargs):
    def choice(self,mode=0,slience=False):
        index = self._auto_choice(mode,slience)
        return torch.cuda.device(index)