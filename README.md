# A manager for NVIDIA card

A GPU devices manager to choice freest gpu.  Forked from https://github.com/QuantumLiu/tf_gpu_manager. 

## Principle

Use ` nvidia-smi --query-gpu={...} --format=csv,noheader` to gather information about current GPU status. Parse it and select GPU according to different rules by returning `with *.device('/gpu:X')`. 

There 3 rules to select GPU (specified by calling `auto_choice(mode_code)`):

0. According to memory on card;


1. According to free memory on card;
2. According to power ratio.

## How to run 

### Brief

1. First copy *GPUmanager* folder to your work folder;
2. import `tfGPUmanager` or `torchGPUmanager` according to your code;
3. Initialize GPUManager using `gm = tfGPUManager()` or `gm = torchGPUManager()`;
4. (For tensorflow,) Use `gm.sess` as your tensorflow session; Before your `sess.run` add `with gm.choice():`;
5. (For pytorch,) Just use `with gm.choice():` before your code.

It looks like this (for tensorflow):

```python
from GPUmanager import tfGPUmanager
import tensorflow as tf
...
gm = tfGPUmanager()
sess = gm.sess
...
with gm.choice():
  sess.run(...)
...
```

It looks like this (for pytorch):

```python
from GPUmanager import torchGPUmanager
import torch
...
gm = torchGPUmanager()
...
CPU = gm.choice():
XXXX.cuda(GPU)
...
```

### Advanced 

You can also ask for multiply cards at a time. This can be done by callin g`give_choices` like this:

```python
GPUManager.give_choices(0,3,slience=False,excludeUsed=True) # this will give you 3 freest cards.
```

 The first parameter 0 is specify in which mode should GPUmanager pick cards, and the second parameter is the number of cards asked.

The `slience` means if GPUmanager give feedbacks, is this is `True`, GPUmanager will not give feedbacks.

And for `excludeUsed`, it means if GPUmanager reuse already specified cards. If this is `True`, these cards been given will not be given again. But you can also manually specify used cards to be reused by calling `include`  method. An example can better explain this mechanism:

```python
# Suppose we have 8 cards.
GPUmanager = torchGPUManager() # the same for tfGPUmanager()
GPUmanager.exclude([0,1]) # this will exclude cards number 0,1 from being used.
print(GPUmanager.give(0,excludeUsed=True)) # this will give you 1 cards to use
print(GPUmanager.give_choices(0,3,excludeUsed=True)) # this will give you 3 cards to use
GPUmanager.include([5]) # reuse no.5 card, if there is no this line, the following line will give a error.
print(GPUmanager.give_choices(1,3,excludeUsed=True)) # this will give you 3 cards to use
GPUmanager.include([0,1,4]) # reuse 3 more cards, if there is no this line, the following line will give a error.
print(GPUmanager.give_choices(2,3,excludeUsed=True)) # this will give you 3 cards to use
```



### In detail

There are mainly two ways to customize it

1. How a card is selected:

   There are three mode as mentioned before, they are

   | mode | sort by                  |
   | ---- | ------------------------ |
   | 0    | largest free memory      |
   | 1    | highest free memory rate |
   | 2    | power                    |

   by default, it will use mode 0, you can customize this by calling `auto_choice` with parameters like:

   `auto_choice(mode=2)` for select according to power.

   Then you can also implement your own mehtod by insert a `_sort_by_XXX` function and set corresponding mode.

2. How to customize Session (for tensorflow):

   When init `GPUManager` you can calling `gm = manager.GPUManager(session=YOUR_SESSION)`, the session your passing in will be configured to use multiply GPU properly. Also, you can refuse GPUManager's session by calling `gm = manager.GPUManager(initSession=False)` , and init your own session somewhere else, and be sure to remember set `allow_soft_placement=True` and `gpu_options.allow_growth=True`.

## Q&A

1. Q: Why am I getting error saying `...because no supported kernel for GPU devices is available.`?

   A: Use default session comes with GPUManager, or add `allow_soft_placement=True` in your config. 

2. Q: Why my session occupying all the memory (using tensorflow)?

   A:  Use default session comes with GPUManager, or add `gpu_options.allow_growth=True` in your config.

3. Q: Why am I getting error saying `Not enough GPU available` ?

   A: You are using `excludeUsed` mode in which used cards are recorded and will not be used again. You can specifiy `excludeUsed=False` (this is by default) when calling `give` `give_choice`  and`choice`. Or you can include some cards in use by calling`include([Nums, of, cards, to, use])`.

## Reference

1. https://zhuanlan.zhihu.com/p/28690706