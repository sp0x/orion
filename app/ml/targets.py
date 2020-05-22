import logging
import time
from datetime import datetime, timedelta
import pandas as pd


def get_target_column_name(target):
    if target.startswith('target_value_'):
        return target
    else:
        return 'target_value_{0}'.format(target)


def parse_targets(targets, collections):
    tp = TargetPicker(targets, collections[0])
    types = tp.get_task_types()
    return types


class TargetPicker(object):
    def __init__(self, target_definitions, data_collection):
        """
        :param target_definitions: 
        target_definitions = [
            {
            'column': 'pm10',
            'constraints': [
                {
                    'type': 'time',
                    'key': 'timestamp',
                    'after': {
                        'hours': 1,
                        'days': 1,
                        'seconds': 1
                    }
                },
                { 'key': 'city_ID'},
                { 'key': 'device_id'},
            ]}
        ]
        :param data_collection: The mongodb collection from which to query target values
        """
        self.targets = target_definitions
        self.collection = data_collection
        assert isinstance(self.targets, list)

    def columns(self):
        return [x['column'] for x in self.targets]

    def count(self):
        return len(self.columns())

    def get_task_types(self):
        from ml.optimizers import get_ml_backend
        sample = self.get_sample(100)
        backend = get_ml_backend(None)
        output = {}
        for t in self.columns():
            tmpdf = pd.DataFrame()
            tmpdf[t] = sample[t]
            task_type = backend.get_task_type(tmpdf)
            output[t] = task_type
        return output

    def __parse_time_constraint(self, record, constraint):
        """

        :param record:
        :param constraint:
        :return:
        """
        tskey = constraint['key']
        ts = pd.to_datetime(record[tskey]).to_pydatetime() if isinstance(tskey, str) else record[tskey]
        shift = timedelta()
        time_filter = constraint['after'] if 'after' in constraint else constraint['before']
        sign = 1 if 'after' in constraint else -1
        if 'hours' in time_filter.keys():
            shift += timedelta(hours=sign * float(time_filter['hours']))
        if 'days' in time_filter.keys():
            shift += timedelta(days=sign * float(time_filter['days']))
        if 'seconds' in time_filter.keys():
            shift += timedelta(seconds=sign * float(time_filter['seconds']))
        period_start = ts + shift
        qr_key = '$gte'
        sorting = None
        if 'after' in constraint:
            qr_key = '$gte'
            sorting = (tskey, 1)
        elif 'before' in constraint:
            qr_key = '$lte'
            sorting = (tskey, 1)
        return {
            'key': qr_key,
            'value': period_start,
            'sort': sorting
        }

    def get_sample(self, count):
        from db.data_handlers import MongoBufferedReader
        mdf = MongoBufferedReader()
        targets_sample = mdf.sample(self.collection, self.columns(), count=count)
        return targets_sample

    def get_target_query(self, target, record):
        """

        :param target:
        :param record:
        :return:
        """
        query = {}
        extras = {
            'sorting': []
        }
        assert len(self.targets) > 0
        assert isinstance(target, str)
        target = [x for x in self.targets if x['column'] == target][0]
        for t in target['constraints']:
            if 'type' in t and t['type'] == 'time':
                tsquery = self.__parse_time_constraint(record, t)
                tsf_key = tsquery['key']
                tsf_value = tsquery['value']
                sorting = tsquery['sort']
                query[t['key']] = {
                    tsf_key: tsf_value
                }
                extras['sorting'].append(sorting)
            else:
                if t['key'] not in record:
                    raise Exception("Key {0} not contained in record.".format(t['key']))
                query[t['key']] = record[t['key']]
        return {
            'query': query,
            'extras': extras
        }

    def get_target_values(self, record):
        target_values = dict()
        for target in self.columns():
            record_query = self.get_target_query(target, record)
            query, extras = (record_query['query'], record_query['extras'])  # , timedelta(hours=1))
            if len(query) == 0:
                matching_row = record
            else:
                matching_row = self.collection.find_one(query, sort=extras['sorting'])
            target_values[target] = matching_row[target] if matching_row is not None else record[target]
        return target_values

    def fill_target_values(self, df):
        ix = 0
        start_time = time.time()
        for tgt in self.columns():
            df['target_value_{0}'.format(tgt)] = 0
            ix += 1
        #
        for index, row in df.iterrows():
            t_vals = self.get_target_values(row)
            ix = 0
            for key in t_vals:
                value = t_vals[key]
                target_value_name = get_target_column_name(key)
                row[target_value_name] = value
                df[target_value_name][index] = value
                ix += 1
        elapsed_time = time.time() - start_time
        logging.info("Target loading took " + str(elapsed_time) + "")
        return df

    def __get_from_frame(self, df):
        tgt_cols = self.columns()
        targets_data = df[tgt_cols]
        df.drop(tgt_cols, axis=1, inplace=True)
        return targets_data

    def get_frame(self, source_df=None):
        """
        Gets a frame of target values from a source frame
        :param source_df:
        :return:
        """
        frame = pd.DataFrame()
        if source_df is None:
            self.fill_target_values(frame)
        else:
            frame = self.__get_from_frame(source_df)
        return frame

    def extract_targets_from_features(self, features_dict, drop_current=True):
        """
        Strips the targets from a dictionary of DFs
        :param features_dict:
        :return:
        """
        from ml.feature_extraction import get_target_column_name
        output_df = pd.DataFrame()
        for t in [x for x in features_dict if x in self.columns()]:
            features = features_dict[t]
            f_name = get_target_column_name(t)
            output_df[t] = features[f_name]
            if t in features:
                features.drop(t, axis=1, inplace=True)
            if f_name in features:
                features.drop(f_name, axis=1, inplace=True)
        return output_df

    def decode(self, targets_df, data_flags):
        from db.encoding import EncodingHelper
        encoder = EncodingHelper()
        return targets_df
