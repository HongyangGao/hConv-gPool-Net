import os
import tensorpack as tp
from utils.data_util import get_data
from utils import voc_util
from model import Model


class Trainer(object):

    def __init__(self, conf, Net):
        self.conf = conf
        self.Net = Net

    def get_config(self):
        tp.logger.set_logger_dir(self.conf.log_dir, 'd')
        vocab = voc_util.get_vocab(self.conf.vob_dict_path)
        dataset_train = get_data(
            self.conf.data_dir, self.conf.batch,
            vocab, self.conf.POS_filter, Windsize=self.conf.Windsize,
            stride=self.conf.stride, is_train=True,
            nV=self.conf.nV, nF=self.conf.nF)
        steps_per_epoch = dataset_train.size()
        dataset_test = get_data(
            self.conf.data_dir, self.conf.batch,
            vocab, self.conf.POS_filter, Windsize=self.conf.Windsize,
            stride=self.conf.stride, is_train=False,
            nV=self.conf.nV, nF=self.conf.nF)
        drop_schedule = []
        for i, epoch in enumerate(map(int, self.conf.drop_epochs.split(','))):
            drop_schedule.append(
                (epoch, self.conf.init_lr * self.conf.drop_rate**(i+1)))
        return tp.TrainConfig(
            dataflow=dataset_train,
            callbacks=[
                tp.ModelSaver(),
                tp.InferenceRunner(
                    dataset_test,
                    [tp.ScalarStats('cost'), tp.ClassificationError()]),
                tp.ScheduledHyperParamSetter('learning_rate', drop_schedule)],
            model=Model(self.conf, self.Net),
            steps_per_epoch=steps_per_epoch,
            max_epoch=self.conf.max_epoch,
        )

    def train(self):
        config = self.get_config()
        if self.conf.load_step:
            config.session_init = tp.SaverRestore(self.conf.load_step)
        gpus = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        tp.launch_train_with_config(config, tp.SyncMultiGPUTrainer(gpus))
