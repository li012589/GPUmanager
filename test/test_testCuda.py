import os
import sys
sys.path.append(os.getcwd())

from utils.testCuda import check_gpus

def test_check_gpus():
    ret = check_gpus()
    print(ret)

if __name__ == "__main__":
    test_check_gpus()