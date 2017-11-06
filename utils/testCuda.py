import os

def check_gpus():
    '''
    GPU available check
    reference : http://feisky.xyz/machine-learning/tensorflow/gpu_list.html
    '''
    try:
        first_gpus = os.popen('nvidia-smi --query-gpu=index --format=csv,noheader').readlines()[0].strip()
        if not first_gpus=='0':
            print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
            return False
        elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
            print("'nvidia-smi' tool not found.")
            return False
    except:
           print("Command line error")
           return False
    return True