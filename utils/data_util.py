import csv
import tensorpack as tp
import numpy as np
from random import shuffle
import utils.voc_util as v_util
import utils.graph_util as g_util


class GraphDataFlow(tp.DataFlow):

    def __init__(self, data_dir, vob, POS_filter, Windsize, stride,
                 is_train, nV, nF):
        self.is_train, self.nV, self.nF = is_train, nV, nF
        self.data_path = data_dir
        tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR',
                'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.POS_filter = tags[:POS_filter]
        self.vobs_dict = vob
        self.Windsize = Windsize
        self.stride = stride
        self.data = []
        file_name = 'train.csv' if is_train else 'test.csv'
        with open(self.data_path + file_name, 'rt') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                self.data.append(row)

    def get_data(self):
        if self.is_train:
            shuffle(self.data)
        for row in self.data:
            d = self.pack_row(row)
            if d is None:
                continue
            yield d

    def pack_row(self, row):
        txt = ' '.join(row[1:])
        tokens = v_util.tokenizer(txt, build_vob=False)
        node_dict, node_list = g_util.get_node_info(
            self.vobs_dict, tokens, self.POS_filter)
        edges_list = g_util.get_edges(
            tokens, self.Windsize, node_dict, self.stride)
        A = g_util.get_Amatrix(node_list, edges_list)
        X = g_util.get_X(self.vobs_dict, node_dict, self.nF)
        y = int(row[0]) - 1
        if X is None:
            return None
        A = A[:self.nV, :self.nV]
        X = X[:self.nV, :]
        if len(A) < self.nV:
            dif = self.nV - len(A)
            A = np.pad(
                A, ((0, dif), (0, dif)), 'constant',
                constant_values=(0, 0))
        word_len = len(X)
        if len(X) < self.nV:
            dif1 = self.nV - len(X)
            dif2 = self.nF - len(X[0])
            X = np.pad(X, ((0, dif1), (0, dif2)), 'constant',
                       constant_values=(0, 0))
        pos_X = self.get_pos_feas(word_len)
        X = np.concatenate([X, pos_X], axis=1)
        return [A, X, y]

    def get_pos_feas(self, word_len):
        x = np.zeros((self.nV, self.nV), dtype=np.float)
        for idx in range(min(word_len, self.nV)):
            x[idx, idx] = 1
        return x

    def size(self):
        return len(self.data)


def get_data(data_dir, batch, vob_dict_path, POS_filter,
             Windsize=3, stride=1, is_train=False, nV=20, nF=300):
    ds = GraphDataFlow(data_dir, vob_dict_path, POS_filter,
                       Windsize, stride, is_train, nV, nF)
    ds = tp.BatchData(ds, batch, remainder=not is_train)
    ds = tp.PrefetchDataZMQ(ds, 10) if is_train else ds
    return ds


if __name__ == '__main__':
    data_dir = '../../data/train.csv'
    vob_dict_path = '../../vob_dict/wiki.simple.vec'
    batch = 10
    POS_filter = 2
    Windsize = 10
    stride = 1
    fre = 5
    is_train = True
    nV = 16
    nF = 300
    data = get_data(
        data_dir, batch, vob_dict_path, POS_filter,
        Windsize, stride, is_train, nV, nF)
    print(data.size())
