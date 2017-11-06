from  manager import GPUManagerTemplate

import tensorflow as tf

class tfGPUmanager(GPUManagerTemplate):
    def __init__(self,qargs=[],sessionargs=None,initSession=None):
        super(tfGPUmanager,self).__init__(qargs)
        if initSession is None:
            if sessionargs is None:
                print("Start session using default args")
                sessionargs = tf.ConfigProto()
            else:
                print("Start session using provided args")
            sessionargs.allow_soft_placement=True
            sessionargs.gpu_options.allow_growth=True
            self.sess = tf.Session(config=sessionargs)
        else:
            print("Using provided session")
            self.sess = initSession
    def choice(self,mode=0,slience=False,excludeUsed=False):
        index = self.give(mode,slience,excludeUsed)
        return tf.device('/gpu:{}'.format(index))

