import glob
import os, sys
import pickle

import numpy as np
from codecs import open as codecs_open
import re
from collections import Counter, defaultdict

UNK_TOKEN = '<unk>'  # unknown word
PAD_TOKEN = '<pad>'  # pad symbol
WS_TOKEN = '<ws>'  # white space (for character embeddings)
RANDOM_SEED = 1234
THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def mkdir_p(path):
    try:
        if os.path.isdir(path):
            return
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


re_html = re.compile('<.*?>')
re_num = re.compile('[0-9]')
re_punct = re.compile('[.,:|/"_\[\]()]')
re_xling_symbols = re.compile('[♫♪%–]')
re_spaces = re.compile(' {2,}')


def sanitize_char(text):
    text = re_html.sub('', text)
    text = re_num.sub('', text)
    text = re_punct.sub('', text)
    text = re_xling_symbols.sub('', text)
    text = re_spaces.sub('', text)
    # text = re.sub('<.*?>', '', text)  # html tags in subtitles
    # text = re.sub('[0-9]', '', text)  # arabic numbers
    # text = re.sub('[.,:|/"_\[\]()]', '', text)  # punctuations
    # text = re.sub('[♫♪%–]', '', text)  # cross-lingual symbols
    # text = re.sub(' {2,}', ' ', text)  # more than two consective spaces
    return text.strip().lower()


def char_tokenizer(text):
    seq = list(sanitize_char(text))
    seq = [(WS_TOKEN if x == ' ' else x) for x in seq]
    return seq


def load_vocab(filename, encoding='utf-8'):
    dic = dict()
    with codecs_open(filename, 'r', encoding=encoding) as f:
        for i, line in enumerate(f.readlines()):
            if len(line.strip('\n')) > 0:
                dic[line.strip('\n')] = i
    return dic


def load_embedding(emb_file, vocab_file, vocab_size, encoding='utf-8'):
    import gensim
    print('Reading pretrained word vectors from file ...')
    word2id = load_vocab(vocab_file)
    word_vecs = gensim.models.KeyedVectors.load_word2vec_format(emb_file, encoding=encoding, binary=False)
    emb_size = word_vecs.syn0.shape[1]
    embedding = np.zeros((vocab_size, emb_size))
    for word, j in word2id:
        j = word2id[word]
        if j < vocab_size:
            if word in word_vecs:
                embedding[j, :] = word_vecs[word]
            else:
                embedding[j, :] = np.random.uniform(-0.25, 0.25, emb_size)
    print('Generated embeddings with shape ' + str(embedding.shape))
    return embedding


def load_unicode_block():
    """
    unicode block name table downloaded from
    https://en.wikipedia.org/wiki/Unicode_block
    """
    ret = []
    path = os.path.join(THIS_DIR, 'unicode_block.tsv')
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip('\n').split('\t', 1)
            m = re.match(r"(U\+[A-F0-9]{4})\t(U\+[A-F0-9]{4})", l[1])
            if m:
                g = m.group(1, 2)
                start = (g[0][2:]).encode('unicode_escape').decode('utf-8')
                end = (g[1][2:]).encode('unicode_escape').decode('utf-8')
                c = re.compile('[%s-%s]' % (start, end))
                ret.append((l[0], c))
    return ret


def save_vocab(data_dir, dic, max_size, encoding='utf-8'):
    # save vocab
    vocab_file = os.path.join(data_dir, 'vocab.txt')
    with codecs_open(vocab_file, 'w', encoding=encoding) as f:
        for char, idx in sorted(dic.items(), key=lambda x: x[1])[:max_size]:
            f.write(char + '\n')

    # save metadata
    unicode_block = load_unicode_block()

    def script(char):
        if char not in [UNK_TOKEN, PAD_TOKEN, WS_TOKEN]:
            for key, c in unicode_block:
                if c.match(char):
                    category = key
                    if category in ['Common', 'Inherited']:
                        category = 'Others'
                    return category
        return 'Others'

    meta_file = os.path.join(data_dir, 'metadata.tsv')
    with codecs_open(meta_file, 'w', encoding=encoding) as f:
        f.write('Char\tScript\n')
        for char, idx in sorted(dic.items(), key=lambda x: x[1])[:max_size]:
            f.write(char + '\t' + script(char) + '\n')


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def latest_file(filepath):
    files = glob.glob(filepath + "*")
    last_file = max(files, key=os.path.getctime)
    return last_file


def load(path):
    out = None
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out


class VocabLoader(object):
    """Load vocabulary"""

    def __init__(self, data_dir):
        self.word2id = None
        self.max_sent_len = None
        self.class_names = None
        self.restore(data_dir)

    def restore(self, data_dir):
        from utils import load
        class_file = os.path.join(data_dir, 'preprocess.cPickle')
        restore_params = load(class_file)
        self.class_names = restore_params['class_names']
        self.max_sent_len = restore_params['max_sent_len']
        print('Loaded target classes (length %d).' % len(self.class_names))

        vocab_file = os.path.join(data_dir, 'vocab.txt')
        self.word2id = load_vocab(vocab_file)
        print('Loaded vocabulary (size %d).' % len(self.word2id))

    def text2id(self, raw_text, auto_trim=True):
        """
        Generate id data from one raw sentence
        """
        if not self.max_sent_len:
            raise Exception('max_sent_len is not set.')
        if not self.word2id:
            raise Exception('word2id is not set.')
        max_sent_len = self.max_sent_len

        toks = char_tokenizer(raw_text)
        toks_len = len(toks)
        pad_left = 0
        pad_right = 0
        if toks_len <= max_sent_len:
            pad_left = int(int((max_sent_len - toks_len)) / int(2))
            pad_right = int(np.ceil((max_sent_len - toks_len) / 2.0))
        else:
            if auto_trim:
                toks = toks[:max_sent_len]
                toks_len = len(toks)
            else:
                return None
        toks_ids = [1 for _ in range(pad_left)] + \
                   [self.word2id[t] if t in self.word2id else 0 for t in toks] + \
                   [1 for _ in range(pad_right)]
        return toks_ids


class TextReader(object):
    """Read raw text"""

    def __init__(self, data_dir, class_names):
        self.data_dir = data_dir
        self.class_names = list(set(class_names))
        self.num_classes = len(set(class_names))
        self.data_files = None
        self.init()

    def init(self):
        if not os.path.exists(self.data_dir):
            sys.exit('Data directory does not exist.')
        self.set_filenames()

    def set_filenames(self):
        data_files = {}
        for f in os.listdir(self.data_dir):
            f = os.path.join(self.data_dir, f)
            if os.path.isfile(f):
                chunks = f.split('.')
                class_name = chunks[-1]
                if class_name in self.class_names:
                    data_files[class_name] = f
        assert data_files
        self.data_files = data_files

    def prepare_dict(self, vocab_size=10000, encoding='utf-8'):
        max_sent_len = 0
        c = Counter()
        # store the preprocessed raw text to avoid cleaning it again
        self.tok_text = defaultdict(list)
        for label in self.data_files:
            f = self.data_files[label]
            with codecs_open(f, 'r', encoding=encoding) as infile:
                for line in infile:
                    toks = char_tokenizer(line)
                    if len(toks) > max_sent_len:
                        max_sent_len = len(toks)
                    for t in toks:
                        c[t] += 1
                    self.tok_text[label].append(' '.join(toks))
        total_words = len(c)
        assert total_words >= vocab_size
        word_list = [p[0] for p in c.most_common(vocab_size - 2)]
        word_list.insert(0, PAD_TOKEN)
        word_list.insert(0, UNK_TOKEN)
        self.word2freq = c
        self.word2id = dict()
        self.max_sent_len = max_sent_len
        for idx, w in enumerate(word_list):
            self.word2id[w] = idx
        save_vocab(self.data_dir, self.word2id, vocab_size)
        print('%d words found in training set. Truncated to vocabulary size %d.' % (total_words, vocab_size))
        print('Max sentence length in data is %d.' % (max_sent_len))
        return

    def generate_id_data(self):
        self.id_text = defaultdict(list)
        for label in self.tok_text:
            sequences = self.tok_text[label]
            for seq in sequences:
                toks = seq.split()
                toks_len = len(toks)
                pad_left = 0
                if toks_len <= self.max_sent_len:
                    pad_left = int((self.max_sent_len - toks_len) / 2)
                    pad_right = int(np.ceil((self.max_sent_len - toks_len) / 2.0))
                else:
                    continue
                toks_ids = [1 for _ in range(pad_left)] \
                           + [self.word2id[t] if t in self.word2id else 0 for t in toks] \
                           + [1 for _ in range(pad_right)]
                self.id_text[label].append(toks_ids)
        return

    def shuffle_and_split(self, test_size=50, shuffle=True):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for label in self.id_text:
            sequences = self.id_text[label]
            length = len(sequences)
            train_size = length - test_size

            if shuffle:
                np.random.seed(RANDOM_SEED)
                permutation = np.random.permutation(length)
                sequences = [sequences[i] for i in permutation]

            # one-hot encoding
            label_id = [0] * self.num_classes
            label_id[self.class_names.index(label)] = 1

            test_x.extend(sequences[train_size:])
            test_y.extend([label_id for _ in range(test_size)])
            train_x.extend(sequences[:train_size])
            train_y.extend([label_id for _ in range(train_size)])

        assert len(train_x) == len(train_y)
        assert len(test_x) == len(test_y)
        train_path = os.path.join(self.data_dir, 'train.cPickle')
        test_path = os.path.join(self.data_dir, 'test.cPickle')
        save((train_x, train_y), train_path)
        save((test_x, test_y), test_path)
        print('Split dataset into train/test set: %d for training, %d for evaluation.' % (len(train_y), len(test_y)))
        return len(train_y), len(test_y)

    def prepare_data(self, vocab_size=10000, test_size=50, shuffle=True):
        # test_size <- per class
        self.prepare_dict(vocab_size)
        self.generate_id_data()
        train_size, test_size = self.shuffle_and_split(test_size, shuffle)
        # test_size <- total
        preprocess_log = {
            'vocab_size': vocab_size,
            'class_names': self.class_names,
            'max_sent_len': self.max_sent_len,
            'test_size': test_size,
            'train_size': train_size
        }
        preprocess_path = os.path.join(self.data_dir, 'preprocess.cPickle')
        save(preprocess_log, preprocess_path)
        return


class DataLoader(object):
    """Load preprocessed data"""

    def __init__(self, data_dir, filename, batch_size=100, shuffle=True):
        from utils import load
        self._x = None
        self._y = None
        self.shuffle = shuffle
        self.load_and_shuffle(data_dir, filename)

        self._pointer = 0

        self._num_examples = len(self._x)
        self.batch_size = batch_size if batch_size > 0 else self._num_examples
        self.num_batch = int(np.ceil(self._num_examples / float(self.batch_size)))
        self.sent_len = len(self._x[0])

        self.num_classes = len(self._y[0])
        self.class_names = load(os.path.join(data_dir, 'preprocess.cPickle'))['class_names']
        assert len(self.class_names) == self.num_classes

        print('Loaded target classes (length %d).' % len(self.class_names))
        print('Loaded data with %d examples. %d examples per batch will be used.' % \
              (self._num_examples, self.batch_size))

    def load_and_shuffle(self, data_dir, filename):
        from utils import load
        _x, _y = load(os.path.join(data_dir, filename))
        assert len(_x) == len(_y)
        if self.shuffle:
            np.random.seed(RANDOM_SEED)
            permutation = np.random.permutation(len(_y))
            _x = np.array(_x)[permutation]
            _y = np.array(_y)[permutation]
        self._x = np.array(_x)
        self._y = np.array(_y)
        return

    def next_batch(self):
        if self.batch_size + self._pointer >= self._num_examples:
            batch_x, batch_y = self._x[self._pointer:], self._y[self._pointer:]
            return batch_x, batch_y
        self._pointer += self.batch_size
        return (self._x[self._pointer - self.batch_size:self._pointer],
                self._y[self._pointer - self.batch_size:self._pointer])

    def reset_pointer(self):
        self._pointer = 0
        if self.shuffle:
            np.random.seed(RANDOM_SEED)
            permutation = np.random.permutation(self._num_examples)
            self._x = self._x[permutation]
            self._y = self._y[permutation]
