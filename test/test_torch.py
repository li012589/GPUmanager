import os
import sys
sys.path.append(os.getcwd())

import torch

from manager_torch import torchGPUmanager

def test_tf_auto_choice():
    t = torchGPUmanager()
    with t.choice():
        x = torch.Tensor(8, 42)
        x = x.cuda()
        print(x)

if __name__ == "__main__":
    test_tf_auto_choice()