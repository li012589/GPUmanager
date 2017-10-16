# A manager for NVIDIA card

A GPU devices manager to choice freest gpu.  Forked from https://github.com/QuantumLiu/tf_gpu_manager. 

## Principle

Use ` nvidia-smi --query-gpu={...} --format=csv,noheader` to gather information about current GPU status. Parse it and select GPU according to different rules by returning `with tf.device('/gpu:X')`. 

There 3 rules to select GPU (specified by calling `auto_choice(mode_code)`):

0. According to memory on card;


1. According to free memory on card;
2. According to power ratio.

## How to run 

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

There are mainly two ways to customize it

1. How select a card:

   There are three mode as mentioned before, they are

   | mode | sort by                  |
   | ---- | ------------------------ |
   | 0    | largest free memory      |
   | 1    | highest free memory rate |
   | 2    | power                    |

   by default, it will use mode 0, you can customize this by calling `auto_choice` with parameters like:

   `auto_choice(mode=2)` for select according to power.

   Then you can also implement your own mehtod by insert a `_sort_by_XXX` function and set corresponding mode.

2. How to customize Session:

   When init `GPUManager` you can calling `gm = manager.GPUManager(session=YOUR_SESSION)`, the session your passing in will be configured to use multiply GPU properly. Also, you can refuse GPUManager's session by calling `gm = manager.GPUManager(initSession=False)` , and init your own session somewhere else, and be sure to remember set `allow_soft_placement=True` and `gpu_options.allow_growth=True`.

## Q&A

1. Q: Why am I getting error saying `...because no supported kernel for GPU devices is available.`?

   A: Use default session comes with GPUManager, or add `allow_soft_placement=True` in your config. 

2. Q Why my session occupying all the memory?

   A:  Use default session comes with GPUManager, or add `gpu_options.allow_growth=True` in your config.

## Reference

1. https://zhuanlan.zhihu.com/p/28690706