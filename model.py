import tensorflow as tf
import tensorpack as tp


class Model(tp.ModelDesc):
    def __init__(self, conf, Net):
        super(Model, self).__init__()
        self.Net = Net
        self.conf = conf

    def inputs(self):
        return [
            tf.placeholder(
                tf.float32, [None, self.conf.nV, self.conf.nV], 'matrix'),
            tf.placeholder(
                tf.float32, [None, self.conf.nV, self.conf.nF+self.conf.nV],
                'feature'),
            tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, matrix, feature, label):
        logits = self.Net(self.conf).inference(matrix, feature)
        self.get_model_cost(logits, label)
        return self.cost

    def get_model_cost(self, logits, label):
        cost = tf.losses.sparse_softmax_cross_entropy(
            labels=label, logits=logits, scope='cross_entropy_loss')

        def prediction_incorrect(logits, label, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, 1))
            return tf.cast(x, tf.float32, name=name)

        wrong = tf.reduce_mean(
            prediction_incorrect(logits, label),
            name='train_error')
        tp.summary.add_moving_summary(wrong)
        wd_cost = tf.multiply(
            1e-4, tp.regularize_cost('.*/weights', tf.nn.l2_loss),
            name='wd_cost')
        tp.summary.add_moving_summary(cost, wd_cost)
        tp.summary.add_param_summary(('.*/kernel', ['histogram']))
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable(
            'learning_rate', initializer=self.conf.init_lr, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.contrib.opt.NadamOptimizer(lr)
