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
    print(t.give_choices(3,0))
    print(t.give_choices(3,1))
    print(t.give_choices(3,2))
    print(t.give_choices(3,3))

if __name__ == "__main__":
    test_template_give()
    test_template_choices()