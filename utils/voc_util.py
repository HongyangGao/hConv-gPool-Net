from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
import csv

lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))


def tokenizer(line, build_vob=False):
    words = []
    new_line = ''
    for c in line:
        new_line += c if c in string.ascii_letters else ' '
    words = [w for w in word_tokenize(new_line.lower()) if len(w) > 2]
    return words


def build_vocab(data_path, vob_path, vocab):
    print("start building vocabulary...")
    un_words = set()
    vocab_dict = {}
    with open(data_path, 'rt') as f:
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for index, row in enumerate(rdr):
            txt = ' '.join(row[1:])
            tokens = tokenizer(txt, build_vob=True)
            for word in tokens:
                if word not in vocab_dict and word in vocab:
                    vocab_dict[word] = {'vec': vocab[word]}
                if word not in vocab and word not in un_words:
                    un_words.add(word)
    print('unkonwn words: ', len(un_words))
    with open('unknown.txt', 'w') as f:
        for word in un_words:
            f.write(word+'\n')
    tokens = dict(nltk.pos_tag(list(vocab_dict.keys())))
    for token, pos in tokens.items():
        vocab_dict[token]['pos'] = pos
    with open(vob_path, 'w') as wf:
        for w, d in vocab_dict.items():
            wf.write(' '.join([w, d['pos'], d['vec']]))
    print("Build %s done!" % vob_path)


def load_vocab(vob_path):
    vocab = {}
    with open(vob_path, 'r') as dic:
        for line in dic:
            word, v = line.split(' ', 1)
            vocab[word] = v
    return vocab


def get_vocab(vocab_path):
    vocab = {}
    with open(vocab_path) as v_file:
        v_file.readline()
        for line in v_file:
            word, pos, v = line.strip().split(' ', 2)
            vocab[word] = {
                'vec': list(map(float, v.split())), 'pos': pos}
    return vocab


if __name__ == '__main__':
    vocab = load_vocab('/tempspace2/hgao/data/fasttext/wiki.en.vec')
    build_vocab('../data/AG/train_full.csv', '../data/AG/ag_fast.vec', vocab)
