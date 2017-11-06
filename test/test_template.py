import os
import sys
sys.path.append(os.getcwd())

from manager import GPUManagerTemplate

def test_template_give():
    t = GPUManagerTemplate()
    print(t.give(0))
    print(t.give(1))
    print(t.give(2))
    print(t.give(3))

def test_template_choices():
    t = GPUManagerTemplate()
    i = [0,1]
    t.exclude(i)
    print(t.give(0,excludeUsed=True))
    print(t.give_choices(0,3,excludeUsed=True))
    t.include([5])
    print(t.give_choices(1,3,excludeUsed=True))
    t.include(i+[4])
    print(t.give_choices(2,3,excludeUsed=True))

if __name__ == "__main__":
    #test_template_give()
    test_template_choices()