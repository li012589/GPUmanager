from  manager import GPUmanager

import tensorflow as tf

class tfGPUmanager(GPUmanager):
    def __init__(self,qargs=[],sessionargs=None,initSession=True):
        super(tfGPUmanager,self).__init__(qargs):
        if initSession is None:
                if sessionargs is None:
                    print("Start session using default args")
                    sessionargs = tf.ConfigProto()
                else:
                    print("Start session using provided args")
                config.allow_soft_placement=True
                config.gpu_options.allow_growth=True
                self.sess = tf.Session(config=sessionargs)
            else:
                print("Using provided session")
                self.sess = initSession
    def choice(self,mode=0,slience=False):
        index = self._auto_choice(mode,slience)
        return tf.device('/gpu:{}'.format(index))

