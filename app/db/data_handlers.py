import threading
from sklearn import preprocessing
import settings
import pandas as pd
from bson import ObjectId
import json
import datetime
from constants import MAX_FILE_SIZE
from db.encoding import EncodingHelper


class MongoDataStream(object):
    def __init__(self, collection, start_date, end_date, chunk_size=10000, max_items=None):
        self.db = settings.get_db()
        self.source = self.db[collection]
        # total number of batches of data in the db/collection
        if not max_items:
            self.len = self.source.find({'Document.g_timestamp': {'$gte': start_date, '$lt': end_date}}).count()
        else:
            self.len = max_items
        self.slices = int(self.len / chunk_size)
        self.data = []  # always keep 2 slices 1 to read from and 1 as a buffer
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.available = True
        self.start = start_date
        self.end = end_date
        self.offset = 0
        self.chunk_size = chunk_size
        self.data = [self.__fetch__() for _ in xrange(2)]
        self.order = []

    def _get_next_(self, offset):
        return self.order[offset:offset + self.chunk_size]

    def reset_stream(self):
        with self.lock:
            self.offset = 0
            self.slices = int(self.len / self.chunk_size)

    def __fetch__(self):
        with self.lock:
            offset = self.offset
        ids = self._get_next_(offset)
        data = self.source.find({'Document.g_timestamp': {'$gte': self.start, '$lt': self.end},
                                 'Document.uuid': {"$in": ids}})
        if self.slices == 0: return
        with self.lock:
            self.data.append(data)
            self.slices -= 1
            self.offset += self.chunk_size
            self.available = True
        with self.cond:
            self.cond.notifyAll()

    def __pre_load__(self):
        t = threading.Thread(target=self.__fetch__)
        t.daemon = True
        t.start()

    def get_doc_ids(self):
        """Retrieves the ids of all users in the db with their status(paid/unpaid)"""
        ids = self.source.find({"UserId": "123123123"}, {"_id": 0, "Document.uuid": 1, "Document.is_paying": 1})
        payment_stats = self.source.find()  # insert the query here
        return ids, payment_stats

    def read(self):
        while len(self.data) > 0:
            with self.lock:
                t_avl = self.available or self.slices == 0
            while not t_avl:
                with self.cond:
                    self.cond.wait(1)
                with self.lock:
                    t_avl = self.available or self.slices == 0
            with self.lock:
                d = self.data.pop()
                self.available = False
            yield d
            self.__pre_load__()
        return


class MongoDataStreamReader(object):
    def __init__(self, stream, features, normalize=False):
        self.stream = stream
        self.features = features
        self.normalize = normalize

    def set_order(self, ids):
        self.stream.order = ids

    def reset_cursor(self):
        self.stream.reset_stream()

    def get_training_set(self):
        return self.stream.get_doc_ids()

    def set_normalize(self, value):
        self.normalize = value

    def read(self):
        data = self.stream.read()
        for d in data:
            doc = []
            if d is None:
                return
            for dd in d:
                tmp = dd['Document']
                for f in self.features:
                    doc.append(tmp[f])
                if self.normalize:
                    yield preprocessing.normalize(doc)
                else:
                    yield doc


class MongoReader(object):
    def __init__(self):
        self.db = settings.get_db()
        self.encoding_helper = EncodingHelper()

    def read(self, collection, start_date, end_date,
             aggregate_steps=None,
             filter_nulls=True,
             exclude_id=False,
             fields=None,
             sample=True,
             limit=True):
        # TODO: simplify this to 1 query
        # Lets assume that all the docs look alike, filter so that all fields MUST have a value
        fields = fields if fields is not None else self.db[collection].find_one()
        doc_filter = {}
        if fields != None:
            for f in fields:
                key = f['name']
                doc_filter[key] = {
                    '$nin': [None, 'NULL', 'null', ''],
                    '$exists': True
                }
        pipeline = []
        if (filter_nulls):
            pipeline.append({'$match': doc_filter})
        if exclude_id:
            pipeline.append({"$project": {"_id": 0}})

        if aggregate_steps != None:
            pipeline = pipeline + aggregate_steps
        output = self.db[collection].aggregate(pipeline)
        return output

    def read_as_pandas_frame(self, collection, start_date, end_date,
                             aggregate_steps=None,
                             filter_nulls=True,
                             exclude_id=False,
                             fields=None,
                             sample=False,
                             limit=True):
        data = self.read(collection, start_date, end_date, aggregate_steps=aggregate_steps, filter_nulls=filter_nulls,
                         exclude_id=exclude_id,
                         fields=fields, sample=sample, limit=limit)
        data = list(data)
        df = pd.DataFrame(data)
        if fields is not None:
            for f in fields:
                if 'encoding' not in f:
                    continue
                df = self.encoding_helper.encode_frame_field(df, f)
        return df

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.db.logout()


class MongoBufferedReader(object):
    def __init__(self):
        self.db = settings.get_db()
        self.encoding_helper = EncodingHelper()

    def sample(self, collection, columns, count=100):
        columns = columns if columns is not None else [x for x in self.db[collection].find_one()]
        doc_filter = {}
        if columns is not None:
            for fname in columns:
                key = fname
                doc_filter[key] = {
                    '$nin': [None, 'NULL', 'null', ''],
                    '$exists': True
                }
        pipeline = []
        projection = {col: 1 for col in columns}
        query = {}
        for dcf in doc_filter:
            query[dcf] = doc_filter[dcf]
        pipeline.append({'$match': query})
        pipeline.append({'$project': projection})
        pipeline.append({'$sample': {'size': count}})
        output = self.db[collection].aggregate(pipeline)
        data = list(output)  # pandas will not read iterators
        df = pd.DataFrame(data)
        return df

    def read_unlimited_as_pandas_frame(self, collection,
                                       fields=None):
        fields = fields if fields is not None else [x for x in self.db[collection].find_one()]
        doc_filter = {}
        if fields is not None:
            for fname in fields:
                key = fname
                doc_filter[key] = {
                    '$nin': [None, 'NULL', 'null', ''],
                    '$exists': True
                }
        pipeline = []
        query = {}
        for dcf in doc_filter:
            query[dcf] = doc_filter[dcf]
        if len(query) > 0:
            pipeline.append({'$match': query})
        output = self.db[collection].aggregate(pipeline)
        data = list(output)  # pandas will not read iterators
        df = pd.DataFrame(data)
        if fields is not None and len(fields) > 0 and self.encoding_helper is not None:
            for f in fields:
                if 'encoding' not in f:
                    continue
                df = self.encoding_helper.encode_frame_field(df, f)
        return df

    def read(self, collection, start_date, end_date,
             aggregate_steps=None,
             filter_nulls=True,
             exclude_id=False,
             fields=None,
             sample=True,
             limit=True):
        # TODO: simplify this to 1 query
        stats = self.db.command('collstats', collection, scale=1024, size=1, count=1)
        # Lets assume that all the docs look alike, filter so that all fields MUST have a value
        fields = fields if fields is not None else [x for x in self.db[collection].find_one()]
        doc_filter = {}
        if fields != None:
            for f in fields:
                key = f['name']
                doc_filter[key] = {
                    '$nin': [None, 'NULL', 'null', ''],
                    '$exists': True
                }

        size = stats['size']
        count = stats['count']
        mb_size = size / 1024
        max_items = 0
        if mb_size <= MAX_FILE_SIZE:
            max_items = count
        else:
            size_item = count / float(size)
            max_items = int(4096.0 / size_item)
        query = {}
        pipeline = []
        if start_date is not None or end_date is not None:
            query['Document.g_timestamp'] = {}
        if start_date is not None:
            query['Document.g_timestamp']['$gte'] = start_date
        if end_date is not None:
            query['Document.g_timestamp']['$lt'] = end_date
        if filter_nulls:
            for dcf in doc_filter:
                query[dcf] = doc_filter[dcf]
        if len(query) > 0:
            pipeline.append({'$match': query})
        if sample:
            pipeline.append({'$sample': {'size': max_items}})
        if limit:
            pipeline.append({'$limit': max_items})
        if exclude_id:
            pipeline.append({"$project": {"_id": 0}})

        if aggregate_steps is not None:
            pipeline = pipeline + aggregate_steps
        output = self.db[collection].aggregate(pipeline)
        return output

    def read_as_pandas_frame(self, collection, start_date, end_date,
                             aggregate_steps=None,
                             filter_nulls=True,
                             exclude_id=False,
                             fields=None,
                             sample=False,
                             limit=True):
        data = self.read(collection, start_date, end_date, aggregate_steps=aggregate_steps, filter_nulls=filter_nulls,
                         exclude_id=exclude_id,
                         fields=fields, sample=sample, limit=limit)
        data = list(data)  # pandas will not read iterators
        df = pd.DataFrame(data)
        # df = self.encoding_helper.decode_raw(df, {
        #    'fields': fields
        # })
        return df

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.db.logout()
