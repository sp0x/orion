from utils import flatten
import pandas as pd
import numpy as np
from dateutil.parser import parse
import os


def timeify(obj, col):
    obj[col] = parse(obj[col])
    return obj


def get_data_flags():
    fields = None
    flags = {
        'collection': {
            'timestamp': 'timestamp',
            'index': None,
            'key': 'collectionName'
        },
        'fields': fields
    }
    return flags


def get_targets(with_constraints=False):
    constraints = [
        {
            'type': 'time',
            'key': 'timestamp',
            'after': {
                'hours': 1,
                'days': 1,
                'seconds': 1
            }
        },
        {'key': 'city_id'},
        {'key': 'device_ID'},
    ]
    targets = [
                  {
                      'column': 'pm10',
                  }
              ],
    if with_constraints:
        targets[0]['constraints'] = constraints
    return targets


def load_temp_module(src):
    import tempfile
    import ntpath
    from utils import load_module
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.file.write(src.encode('utf-8'))
    temp.file.flush()
    module = load_module(ntpath.basename(temp.name), temp.name)
    return module, temp


def extract_keras_connections(layers):
    gen_cons = {}
    for i in range(len(layers)):
        layer = layers[i]
        in_nodes = filter(lambda x: type(x) is str, flatten(map(lambda x: x[0], layer['inbound_nodes'])))
        if layer['name'] in gen_cons:
            gen_cons.get(layer['name'], {}).update({'in': in_nodes})
        else:
            gen_cons[layer['name']] = {'in': in_nodes, 'out': []}
        for n in in_nodes:
            out = gen_cons.get(n).get('out', [])
            out.append(layer['name'])
            gen_cons[n]['out'] = out
    return gen_cons


def get_testfile(filename):
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, '../data/test_fixtures/', filename)


def read_testfile(filename):
    full_path = get_testfile(filename)
    with open(full_path, 'r') as file_h:
        return file_h.read()


def get_df(filename):
    path = get_testfile(filename)
    df = pd.read_csv(path)
    return df


def get_base_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.realpath(os.path.join(dir_path, '..', '..', 'experiments'))
    return path


def fill_db(db, collectionName, csv_file, formatter=None, cnt=None):
    csv_file = '../data/test_fixtures/' + csv_file
    frame = pd.read_csv(csv_file)
    col = db[collectionName]
    for i, row in frame.iterrows():
        if cnt is not None and cnt <= i:
            break
        to_dict = row.to_dict()
        if formatter is not None:
            to_dict = formatter(to_dict)
        col.insert(to_dict)
    return col


def generate_language_df(test_column, rows=50):
    import nltk
    import random
    random.seed(55)
    from nltk.corpus import udhr
    from timeit import default_timer as timer
    languages = udhr._fileids
    sents_eng = udhr.sents('English-Latin1')
    sents_bg = udhr.sents('Bulgarian_Balgarski-UTF8')
    sents_ger = udhr.sents('German_Deutsch-Latin1')
    cnt_en = int(rows * 0.4)
    cnt_bg = int(rows * 0.4)
    cnt_de = int(rows * 0.2)

    df = pd.DataFrame(np.random.randn(rows, 4), columns=['A', 'B', 'C', test_column])
    df.set_index('A')
    tcol = df[test_column]
    nums = []
    for i in range(cnt_en):
        rnd_sent = ' '.join(random.choice(sents_eng))
        nums.append(rnd_sent)
    for i in range(cnt_bg):
        rnd_sent = ' '.join(random.choice(sents_bg))
        nums.append(rnd_sent)
    for i in range(cnt_de):
        rnd_sent = ' '.join(random.choice(sents_ger))
        nums.append(rnd_sent)
    df.loc[:, test_column] = nums
    return df


class MockMongoBufferedReader(object):
    def __init__(self, filepath='../data/test_fixtures/feature_test.csv', ids=['_id'], limit=None):
        frame = pd.read_csv(filepath)
        if limit is not None:
            frame = frame.head(limit)
        self.frame = frame.sample()
        for i in ids:
            self.frame[i] = range(len(self.frame))

    def read_as_pandas_frame(self, collection, start_date, end_date,
                             aggregate_steps=None,
                             filter_nulls=True,
                             exclude_id=False,
                             fields=None,
                             sample=False,
                             limit=True):
        return self.frame

    def close(self):
        return
