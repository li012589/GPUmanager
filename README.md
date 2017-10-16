# A manager for NVIDIA card

A GPU devices manager to choice freest gpu.  Forked from https://github.com/QuantumLiu/tf_gpu_manager. 

## Principle

Use ` nvidia-smi --query-gpu={...} --format=csv,noheader` to gather information about current GPU status. Parse it and select GPU according to different rules by returning `with tf.device('/gpu:X')`. 

There 3 rules to select GPU (specified by calling `auto_choice(mode_code)`):

0. According to memory on card;


1. According to free memory on card;
2. According to power ratio.

## How to 

  gm=GPUManager()  
  with gm.auto_choice():  
    blabla
## Reference

1. https://zhuanlan.zhihu.com/p/28690706