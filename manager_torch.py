from  manager import GPUManagerTemplate

import torch

class torchGPUmanager(GPUManagerTemplate):
    def __init__(self,qargs=[]):
        super(torchGPUmanager,self).__init__(qargs)
    def choice(self,mode=0,slience=False,excludeUsed = False):
        index = self.give(mode,slience,excludeUsed)
        return torch.cuda.device(index)