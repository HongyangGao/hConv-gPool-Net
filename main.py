import os
import tensorflow as tf
from trainer import Trainer
from network import DenseNet


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_epoch', 60, 'network depth')
    flags.DEFINE_float('init_lr', 0.001, 'initial learning rate')
    flags.DEFINE_float('drop_rate', 0.1, 'drop learning rate rate')
    flags.DEFINE_string('drop_epochs', '30, 50', 'drop epochs')
    flags.DEFINE_string('data_format', 'NCHW', 'data format for training')
    # data
    flags.DEFINE_string('data_dir', 'data/AG/', 'data directory')
    flags.DEFINE_integer('batch', 256, 'batch size')
    flags.DEFINE_integer('nV', 78, 'max number of nodes in graph')
    flags.DEFINE_integer('nF', 300, 'feature number of nodes in graph')
    flags.DEFINE_integer('class_num', 4, 'output class number')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('reload_step', '', 'Reload step to continue training')
    flags.DEFINE_string('test_step', '', 'Test or predict model at this step')
    # network architecture
    flags.DEFINE_integer('ch_num', 256, 'channel num')
    flags.DEFINE_integer('l_num', 3, 'layer num in network')
    flags.DEFINE_float('rate', 0.45, 'dropout rate')
    # text data prep-recess
    flags.DEFINE_string('vob_dict_path', 'data/AG/ag_fast.vec', '')
    flags.DEFINE_integer(
        'POS_filter', 13, 'POS, 4: n. 7: n&adj. 13: n&adj&v.')
    flags.DEFINE_integer(
        'Windsize', 4, 'Windsize for finding a edge between two words.')
    flags.DEFINE_integer('stride', 1, 'Window moving stride.')

    return flags.FLAGS


def main(_):
    Trainer(configure(), DenseNet).train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    tf.app.run()
