import os
from utils import check_gpus

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

class GPUManagerTemplate():
    def __init__(self,qargs=[]):
        assert check_gpus()
        self.qargs=qargs
        self.gpus=query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified']=False
        self.gpu_num=len(self.gpus)

    def _if_all_specified(self):
        remain = 0
        for item in self.gpus:
            if not item['specified']:
                remain += 1
        return remain
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

    def _auto_choice(self,mode=0,num = 1,slience=False,excludeUsed = False):
        remain = self._if_all_specified()
        if remain < num:
            raise ValueError("Not enough GPU available")

        for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
            old_infos.update(new_infos)
        if excludeUsed:
            unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']]
        else:
            unspecified_gpus=self.gpus
        if mode==0:
            if not slience:
                print('Choosing the GPU device has largest free memory...')
            chosen_gpu=self._sort_by_memory(unspecified_gpus,True)[:num]
        elif mode==1:
            if not slience:
                print('Choosing the GPU device has highest free memory rate...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)[:num]
        elif mode==2:
            if not slience:
                print('Choosing the GPU device by power...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)[:num]
        else:
            if not slience:
                print('Given an unaviliable mode,will be chosen by memory')
            chosen_gpu=self._sort_by_memory(unspecified_gpus)[:num]

        index = []
        for item in chosen_gpu:
            item['specified']=True
            index.append(int(item['index']))
            if not slience:
                print('Using GPU {i}:\n{info}'.format(i=index,info='\n'.join([str(k)+':'+str(v) for k,v in item.items()])))
        return index

    def exclude(self,index):
        for i in index:
            self.gpus[i]['specified'] = True
    def include(self,index):
        for i in index:
            self.gpus[i]['specified'] = False
    def give(self,mode=0,slience=False,excludeUsed = False):
        index = self._auto_choice(mode,1,slience,excludeUsed)
        return index[0]
    def give_choices(self,mode=0,num=1,slience=True,excludeUsed=False):
        index = self._auto_choice(mode,num,slience,excludeUsed)
        return index

if __name__ == "__main__":
    t = GPUManagerTemplate()
    i = [0,1]
    t.exclude(i)
    print(t.give(0,excludeUsed=True))
    print(t.give_choices(0,3,excludeUsed=True))
    t.include([5])
    print(t.give_choices(3,3,excludeUsed=True))
    t.include(i+[4])
    print(t.give_choices(3,3,excludeUsed=True))