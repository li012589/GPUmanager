# A manager for NVIDIA card

A GPU devices manager to choice freest gpu.  Forked from https://github.com/QuantumLiu/tf_gpu_manager. 

## Principle

Use ` nvidia-smi --query-gpu={...} --format=csv,noheader` to gather information about current GPU status. Parse it and select GPU according to different rules by returning `with tf.device('/gpu:X')`. 

There 3 rules to select GPU (specified by calling `auto_choice(mode_code)`):

0. According to memory on card;


1. According to free memory on card;
2. According to power ratio.

## How to   

### Briefly

1. First copy *manager.py* to your work folder;
2. import *manager.py* at your code;
3. Initialize GPUManager using `gm = manager.GPUManager()`;
4. Use `gm.sess` as your tensorflow session;
5. Before your `sess.run` add `with gm.auto_choice():`;

It looks like this:

```python
import manager
import tensorflow as tf
...
gm = manager.GPUManger()
sess = gm.sess
...
with gm.auto_choice():
  sess.run(...)
...
```

### In detail

There are mainly two way to customize it

1. ​

## Reference

1. https://zhuanlan.zhihu.com/p/28690706