# coding=utf-8

from datetime import datetime, timedelta
from ml import feature_functions
import numpy as np
import pandas as pd
import settings
import time
from db.encoding import EncodingHelper, __decode_binary_array

encoder = EncodingHelper()


def encode_feature(ftr):
    import bson
    if isinstance(ftr, np.int64):
        output = bson.int64.Int64(ftr)
    elif isinstance(ftr, bool):
        output = 1 if ftr else 0
    elif isinstance(ftr, list):
        output = np.average(ftr) if len(ftr) > 0 else 0
        if isinstance(output, bool):
            output = 1 if output else 0
    else:
        output = ftr
    return output


def bin_to_int(b):
    b = b[1:]
    val = int(b, 2)
    return val


def flatten_feature_values(ftrs):
    for f in ftrs:
        feature_val = ftrs[f]
        if isinstance(feature_val, list):
            if isinstance(feature_val[0], str):

                unique_chars = ''.join(set(feature_val[0]))
                if unique_chars == '01' or unique_chars == '10':
                    feature_val = __decode_binary_array(feature_val)
            elif isinstance(feature_val[0], pd.Timestamp):
                sec_array = [time.mktime(x.to_pydatetime().timetuple()) for x in feature_val]
                feature_val = np.average(sec_array)
            elif isinstance(feature_val[0], datetime):
                sec_array = [time.mktime(x.timetuple()) for x in feature_val]
                feature_val = np.average(sec_array)
            else:
                feature_val = np.average(feature_val)

        ftrs[f] = feature_val
    return ftrs


class DataConsumer:
    def __init__(self, targets, feature_definitions, index='id', timestamp_column='timestamp', grouping_function=None,
                 db_settings=None, save_to_db=True):
        """

        :param targets: A list of targets as strings
        :param feature_definitions: A list of dicts for features
        :param index: The index column for the data
        :param timestamp_column:
        :param grouping_function: Function to use to aggregate
        :param db_settings:
        """
        self.city_bounds = {}
        self.city_grids = {}
        self.targets = targets
        self.feature_definitions = dict()
        self.timestamp_column = timestamp_column  # self.__get_entity_timecol(entry)
        self.grouping_function = (lambda x: (index, x[index])) if grouping_function is None else grouping_function
        self.current_group = None
        self.previous_group = None
        self.grouping = []
        if feature_definitions is not None:
            self.feature_definitions = feature_definitions
        self.grouping = []
        self.data_collection = db_settings['data_collection'] if db_settings is not None else None
        self.features_collection = db_settings['features_collection'] if db_settings is not None else None
        self.should_extract_features = True
        self.save_to_db = save_to_db
        self.feature_function = None

    def set_should_extract_features(self, t):
        self.should_extract_features = t

    def set_feature_definitions(self, definitions):
        self.feature_definitions = definitions

    def add_feature(self, target, definition):
        if target not in self.feature_definitions:
            self.feature_definitions[target] = set()
        self.feature_definitions[target].append(definition)

    def add(self, element):
        """
        Add data to be processed
        :param element:
        :return:
        """
        self.__on_new_data([element])

    def add_all(self, elements):
        """

        :param elements:
        :return:
        """
        self.__on_new_data(elements)

    def __on_new_data(self, elements):
        """
        Processes and saves the elements.
        :param elements:
        :return:
        """
        from bson.objectid import ObjectId
        collection = self.data_collection
        for i, record in enumerate(elements):
            record = self.format_data(record)
            if '_id' not in record:
                record['_id'] = ObjectId()
            self.__process_entry(i, record)
        collection.insert_many(elements)

    def format_data(self, data):
        """
        Formats the fields in the dict
        :param data: dict
        :return:
        """
        if isinstance(data[self.timestamp_column], int):
            data[self.timestamp_column] = datetime.fromtimestamp(data[self.timestamp_column] / 1000)
        return data

    def __process_entry(self, index, entry):
        features = self._push_features_input(entry)
        # if use_target_picking:
        # self.update_target_value(entry, end_index=index)
        return entry

    def _get_features_for_all_targets(self):
        return self.feature_definitions['common'] if self.feature_definitions is not None \
                                                   and 'common' in self.feature_definitions else list()

    def _set_feature_fields(self, dictx):
        from ml.targets import get_target_column_name
        cnt = 0
        for target in [x for x in self.feature_definitions if x != 'common']:
            ftarget = get_target_column_name(target)
            cnt += len(self.feature_definitions[target])
            if ftarget not in dictx:
                dictx[ftarget] = 0.0
        return cnt

    def _push_features_input(self, entry, save_group=True):
        """

        :param entry:
        :param save_group:
        :return:
        """
        from ml.targets import get_target_column_name
        if not self.should_extract_features:
            return None
        group = self.get_entry_group(entry)
        ftr_cnt = len(self._get_features_for_all_targets())
        ftr_cnt += self._set_feature_fields(entry)
        if ftr_cnt == 0 and self.feature_function is None:
            return None

        if self.current_group is None:
            self.current_group = group
            self.grouping.append(entry)
        elif self.current_group != group or len(group) == 0 and len(self.current_group) == 0:
            # Group has changed, extract features for all previous entries with all targets
            feature_values = self.extract_group_features(save_group)
            self.previous_group = self.current_group
            self.current_group = group
            # Clear previous grouped entries
            self.grouping = []
            self.grouping.append(entry)
            return feature_values
        else:
            self.grouping.append(entry)
        return None

    def complete(self):
        output = self.extract_group_features(save_group=False)
        return output

    def extract_group_features(self, save_group):
        feature_values = dict()
        # Unique target features
        for target in [x for x in self.feature_definitions if x != 'common']:
            feature_defs = self.feature_definitions[target]
            # target_features = dict()
            for fdef in feature_defs:
                feature_value = self.calc_feature(fdef)
                f_name = fdef['title'].replace('.', '．').encode('utf-8', 'strict').decode("utf-8")
                feature_values[f_name] = feature_value

        if 'common' in self.feature_definitions:
            # Common features
            for feature_def in self.feature_definitions['common']:
                feature_value = self.calc_feature(feature_def)
                f_name = feature_def['title'].replace('.', '．').encode('utf-8', 'strict').decode("utf-8")
                feature_values[f_name] = feature_value

        if self.feature_function is not None:
            feature_values = self.feature_function(self.grouping)
            feature_values = flatten_feature_values(feature_values)

        if not self.save_to_db:
            return feature_values

        feature_values['_did'] = [doc['_id'] for doc in self.grouping]
        if save_group:
            self.save_features_group(self.current_group, feature_values)
        return feature_values

    def save_features_group(self, group, features):
        bulk_collection = []
        group_dict = dict()
        if isinstance(group, list):
            for gr_by in group:
                group_dict[gr_by] = group[gr_by]
        elif isinstance(group, dict):
            group_dict = group
        else:
            group_dict[group[0]] = group[1]
        grouping_first_elem = self.grouping[0]
        grouping_starting_ts = grouping_first_elem[self.timestamp_column]
        # Use only first target
        # for target in targets_with_features:
        targets = self.targets
        bulk_entry = {
            '_meta': {
                '_t': targets, '_g': group_dict
            },
            '_did': features['_did']
        }
        del features['_did']
        if grouping_starting_ts is not None:
            bulk_entry['_ts'] = grouping_starting_ts
        for ftr in features:
            bulk_entry[ftr] = encode_feature(features[ftr])
        bulk_collection.append(bulk_entry)
        insert_result = self.features_collection.insert_many(bulk_collection)
        return insert_result

    def calc_feature(self, fdef):
        feature_val = feature_functions.resolve(fdef, self.grouping)
        if isinstance(feature_val, list):
            feature_val = np.average(feature_val)
        return feature_val

    def calculate_df(self, df):
        """

        :param df:
        :return:
        """

        def features_gen():
            for i, row in df.iterrows():
                ret = self._push_features_input(row.to_dict(), save_group=False)
                if ret is not None:
                    yield ret
            last_feature = self.complete()
            if last_feature is not None:
                yield last_feature

        features = list(features_gen())
        features_df = pd.DataFrame(features)
        if '_did' in features_df:
            features_df.drop('_did', axis=1, inplace=True)
        return features_df

    def get_single_target_features(self, df, target):
        from bson.objectid import ObjectId
        from ml.cleanup import cleanup_premade_features
        ids = df['_id']
        query = {
            '_did': {
                '$in': []
            },
            '_meta._t': {
                '$in': [target]
            }
        }
        for id in ids:
            query['_did']['$in'].append(ObjectId(id))
        matching_features = list(self.features_collection.find(query, {
            "_id": 0,
            "_did": 0,
            "_g": 0,
            "_t": 0
        }))
        matching_features = cleanup_premade_features(matching_features)
        df = pd.DataFrame(matching_features)
        df_size = len(df)
        return df

    def get_features(self, df, target_names):
        from bson.objectid import ObjectId
        from ml.cleanup import cleanup_premade_features
        ids = df['_id']
        query = {
            '_did': {
                '$in': []
            },
            '_meta._t': {
                '$all': target_names
            }
        }
        for id in ids:
            query['_did']['$in'].append(ObjectId(id))
        matching_features = list(self.features_collection.find(query, {
            "_id": 0,
            "_did": 0,
            "_meta": 0,
        }))
        matching_features = cleanup_premade_features(matching_features)
        output_df = pd.DataFrame(matching_features)
        # Calculate features if we don`t have them
        if len(output_df) == 0:
            output_df = self.calculate_df(df)
        return output_df

    def get_entry_group(self, entry):
        g = self.grouping_function(entry)
        return g

    def set_feature_function(self, feature_function):
        self.feature_function = feature_function
