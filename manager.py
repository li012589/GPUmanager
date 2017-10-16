import os
import tensorflow as tf

def check_gpus():
    '''
    GPU available check
    reference : http://feisky.xyz/machine-learning/tensorflow/gpu_list.html
    '''
    first_gpus = os.popen('nvidia-smi --query-gpu=index --format=csv,noheader').readlines()[0].strip()
    if not first_gpus=='0':
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True

if check_gpus():
    def parse(line,qargs):
        numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
        power_manage_enable=lambda v:(not 'Not Support' in v)
        to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))
        process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

    def query_gpu(qargs=[]):
        qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [parse(line,qargs) for line in results]

    def by_power(d):
        power_infos=(d['power.draw'],d['power.limit'])
        if any(v==1 for v in power_infos):
            print('Power management unable for GPU {}'.format(d['index']))
            return 1
        return float(d['power.draw'])/d['power.limit']

    class GPUManager():
        def __init__(self,qargs=[],sessionargs=None,initSession=True):
            '''
            '''
            self.qargs=qargs
            self.gpus=query_gpu(qargs)
            for gpu in self.gpus:
                gpu['specified']=False
            self.gpu_num=len(self.gpus)
            if initSession:
                if sessionargs is None:
                    print("Start session using default args")
                    config = tf.ConfigProto()
                else:
                    print("Start session using provided args")
                config.allow_soft_placement=True
                config.gpu_options.allow_growth=True
                self.sess = tf.Session(config=config)
            else:
                pass

        def _sort_by_memory(self,gpus,by_size=False):
            if by_size:
                print('Sorted by free memory size')
                return sorted(gpus,key=lambda d:d['memory.free'],reverse=True)
            else:
                print('Sorted by free memory rate')
                return sorted(gpus,key=lambda d:float(d['memory.free'])/ d['memory.total'],reverse=True)

        def _sort_by_power(self,gpus):
            return sorted(gpus,key=by_power)

        def _sort_by_custom(self,gpus,key,reverse=False,qargs=[]):
            if isinstance(key,str) and (key in qargs):
                return sorted(gpus,key=lambda d:d[key],reverse=reverse)
            if isinstance(key,type(lambda a:a)):
                return sorted(gpus,key=key,reverse=reverse)
            raise ValueError("The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

        def auto_choice(self,mode=0):
            for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
                old_infos.update(new_infos)
            unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
            if mode==0:
                print('Choosing the GPU device has largest free memory...')
                chosen_gpu=self._sort_by_memory(unspecified_gpus,True)[0]
            elif mode==1:
                print('Choosing the GPU device has highest free memory rate...')
                chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
            elif mode==2:
                print('Choosing the GPU device by power...')
                chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
            else:
                print('Given an unaviliable mode,will be chosen by memory')
                chosen_gpu=self._sort_by_memory(unspecified_gpus)[0]
            chosen_gpu['specified']=True
            index=chosen_gpu['index']
            print('Using GPU {i}:\n{info}'.format(i=index,info='\n'.join([str(k)+':'+str(v) for k,v in chosen_gpu.items()])))
            return tf.device('/gpu:{}'.format(index))
else:
    raise ImportError('GPU available check failed')
