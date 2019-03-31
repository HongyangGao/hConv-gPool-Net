import tensorflow as tf
from utils import ops


class DenseNet(object):

    def __init__(self, conf):
        self.conf = conf

    def inference(self, matrix, outs):
        outputs = []
        outs = ops.simple_conv(
            matrix, outs, 4*self.conf.ch_num, self.conf.rate,
            'conv0', 3)
        for i in range(self.conf.l_num):
            ratio = 4 // 2**i
            outs = ops.simple_conv(
                matrix, outs, ratio*self.conf.ch_num, self.conf.rate,
                'conv1_%s' % i, 3)
            outputs.append(outs)
            matrix, outs = ops.graph_pool(matrix, outs, 2, 'pool_%s' % i)
        axis_outs = []
        for i, outs in enumerate(outputs):
            outs = tf.reduce_max(outs, axis=1, name='max_pool_%s' % i)
            axis_outs.append(outs)
        outs = tf.concat(axis_outs, axis=1)
        outs = ops.dense(outs, 1024, 'dense1')
        outs = ops.dense(outs, self.conf.class_num, 'dense2')
        return outs
