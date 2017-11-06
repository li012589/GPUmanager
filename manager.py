import os
from .utils.testCuda import check_gpus

class GPUManagerTemplate():
    def __init__(self,qargs=[],sessionargs=None):
        assert check_gpus()
        self.qargs=qargs
        self.gpus=query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified']=False
        self.gpu_num=len(self.gpus)

    def _sort_by_memory(self,gpus,by_size=False):
        if by_size:
            return sorted(gpus,key=lambda d:d['memory.free'],reverse=True)
        else:
            return sorted(gpus,key=lambda d:float(d['memory.free'])/ d['memory.total'],reverse=True)

    def _sort_by_power(self,gpus):
        return sorted(gpus,key=by_power)

    def _sort_by_custom(self,gpus,key,reverse=False,qargs=[]):
        if isinstance(key,str) and (key in qargs):
            return sorted(gpus,key=lambda d:d[key],reverse=reverse)
        if isinstance(key,type(lambda a:a)):
            return sorted(gpus,key=key,reverse=reverse)
        raise ValueError("The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

    def _auto_choice(self,mode=0,slience=False):
        for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
        if mode==0:
            if not slience:
                print('Choosing the GPU device has largest free memory...')
            chosen_gpu=self._sort_by_memory(unspecified_gpus,True)[0]
        elif mode==1:
            if not slience:
                print('Choosing the GPU device has highest free memory rate...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
        elif mode==2:
            if not slience:
                print('Choosing the GPU device by power...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
        else:
            if not slience:
                print('Given an unaviliable mode,will be chosen by memory')
            chosen_gpu=self._sort_by_memory(unspecified_gpus)[0]
        chosen_gpu['specified']=True
        index=chosen_gpu['index']
        print('Using GPU {i}:\n{info}'.format(i=index,info='\n'.join([str(k)+':'+str(v) for k,v in chosen_gpu.items()])))
        return index
    def auto_choice(self,mode=0,slience=False):
        indec = self._auto_choice(mode,slience)
        return tf.device('/gpu:{}'.format(index))
    def give_choices(self,num=1,mode=0,slience=False):
        index = []
        for i in range(num):
            tmp = self._auto_choice(mode=0,slience)
            index.append(tmp)
        return tmp

else:
    raise ImportError('GPU available check failed')
