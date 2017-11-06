import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf

from manager_tf import tfGPUmanager

def test_tf_auto_choice():
    t = tfGPUmanager()
    sess = t.sess
    with tfGPUmanager.choice():
        node1 = tf.constant(3.0, dtype=tf.float32)
        node2 = tf.constant(4.0)
        print(sess.run([node1, node2]))

if __name__ == "__main__":
    test_tf_auto_choice()