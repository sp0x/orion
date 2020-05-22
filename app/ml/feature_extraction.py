import featuretools as ft
from dotmap import DotMap
import pandas as pd
import numpy as np
import logging
import traceback
import os
import envir
import time

from code_generation import get_template_file
from ml.targets import get_target_column_name
from ml.feature_functions import gen_feature_expr
from utils import update_lines
import settings

log = logging.getLogger(__name__)
# hdlr = logging.FileHandler(app.settings.get_log_file(__name__))
# log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# hdlr.setFormatter(log_formatter)
# log.addHandler(hdlr)
log.setLevel(logging.DEBUG)


def create_target_columns(df, targets):
    ix = 0
    for tgt in targets:
        df['target_value_{0}'.format(tgt)] = 0
        ix += 1
    return df


def format_feature_sets(feature_defs, timestamp_column):
    """

    :param feature_defs:  Feature definitions list
    :param timestamp_column: The timestamp column
    :return: A list of feature dicts
    """
    feature_dicts = dict()
    for t in feature_defs:
        fdefs = feature_defs[t]
        feature_dicts[t] = []
        if fdefs is not None:
            for ftr in fdefs:
                ftr_dict = FeaturesBuilder.feature_to_dict(t, ftr, timestamp_column=timestamp_column)
                feature_dicts[t].append(ftr_dict)
                # Remove common features, instead of calculating them multiple times
    common_features = []
    for i, t in enumerate(feature_defs):
        feature_title_set = feature_dicts[t]  # set([x['title'] for x in self.feature_definitions[t]])
        f1_titles = set([x['title'] for x in feature_dicts[t]])
        for i1, t2 in enumerate(feature_dicts):
            if i1 == i:
                continue
            feature_title_set2 = feature_dicts[t2]  # set([x['title'] for x in self.feature_definitions[t2]])
            f2_titles = set([x['title'] for x in feature_title_set2])
            commons = f1_titles.intersection(f2_titles)
            for c in commons:
                cval = next(x for x in feature_title_set2 if x['title'] == c)
                cval['targets'] = [t, t2]
                common_features.append(cval)
                feature_title_set2 = [x for x in feature_title_set2 if x['title'] != c]
                feature_title_set = [x for x in feature_title_set if x['title'] != c]

            feature_dicts[t2] = feature_title_set2
            feature_dicts[t] = feature_title_set
    feature_dicts['common'] = common_features
    return feature_dicts


def format_feature_sets_from_files(client, target, timestamp_column, entity_set):
    import os.path
    from featuretools import load_features
    path = FeaturesBuilder.get_definitions_file(client, target)
    if os.path.isfile(path):
        features = load_features(path, entity_set)
        fdict = {target: features}
        return format_feature_sets(fdict, timestamp_column)
    else:
        return None


class FeaturesBuilder:
    from ml import MlScript

    def __init__(self, data_config, reader, ops='all'):
        """
        //ench
        :param data_config: dict describing of the data to be used
        data_config = {
            'db': MongoDB
            'model_name':'str'
            'collection': {
                'name':'the name of the collection given by the user',
                'key':'the actual name of the collection in mongodb'
                'start':datetime, date to begin the data extraction from
                'end':datetime,   date to read up to
                'index':'index'   what to sort the entities by
                'timestamp': 'timestamp_column'
            },
            'client': 'clientName',
            'verbose': True,
            'export_features': True,
            'relations':[[obj.attr,obj2,attr]],
            'targets':
            {
                'columns': ['pm10', 'pm25'],
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
                ]
            },    # what we're gonna be learning,
            'internal_entities' : [{
                collection: 'samecollection..' id of the collection that the thing is in
                name:'internal_entity1',
                index: 'event_id', in your case event_id
                fields:[type,uuid] - basically the fields it'll use for the new entity
            }],
            'use_feature_generation': True
        }
        :param ops:         list of ops allowed to be used or 'all' for all ops
        Note that each entity set can contain only ONE target.
        """
        from ml.targets import TargetPicker
        from helpers import ClientFs
        self.db = data_config['db']
        dc = DotMap(data_config)
        self._dc = dc
        self.client = dc.client
        self.client_fs = ClientFs(dc.client)
        self._collection = dc.collection
        self.timestamp = dc.collection.timestamp if dc.collection and hasattr(dc.collection, 'timestamp') else None
        self._ops = ops
        self._targets = dc.targets
        self.verbose = dc.verbose
        self.export_features = dc.export_features
        self.use_feature_generation = dc.get('use_feature_generation', default=False)
        self.internal_entities = dc.internal_entities if 'internal_entities' in dc else None
        self._entities = None
        self._target_picker = TargetPicker(self._targets, self._get_data_collection())
        self.feature_definitions = []
        self.data_consumer = None
        self.reader = reader
        self.feature_function = None
        self.grouping_function = None
        self.script = None
        # self._labels = self.__make_labels(1)

    def set_script(self, script: MlScript):
        self.script = script
        self.set_feature_function(script.get_features)

    def set_feature_function(self, fun):
        """
        Set the function that would be used to output features
        :param fun:
        :return:
        """
        self.feature_function = fun

    def set_aggregate_function(self, fun):
        self.grouping_function = fun

    def _get_data_collection(self):
        reader = self.db[self._collection.key]
        return reader

    def set_data_consumer(self, consumer):
        self.data_consumer = consumer

    def create(self, df, target, df_settings):
        """Creates a feature set for a given target
        """
        from sklearn.preprocessing import Imputer
        from ml.cleanup import clear_data_for_features
        # Todo: edit this if you enable feature tools
        output_df = None
        feature_definitions = None
        df.dropna(axis=0, how='any', inplace=True)
        if not self.use_feature_generation:
            # Without features
            # df.drop('timestamp', axis=1, inplace=True)
            # df = clear_data_for_features(df)
            output_df = df
        else:
            full_target_name = get_target_column_name(target)
            df[full_target_name + '_id'] = range(len(df))
            logging.info("Creating features for: " + full_target_name)
            # Create the entity set
            es = self.load_dataset(df, target, settings=df_settings,
                                   cleanup_lambda=lambda x:
                                   clear_data_for_features(x),
                                   fetch_target=False)

            feature_matrix, features = ft.dfs(entityset=es,
                                              target_entity=full_target_name,
                                              max_depth=5,
                                              verbose=self.verbose)

            f_matrix_enc, features_enc = ft.encode_features(feature_matrix, features)
            feature_definitions = features_enc
            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            f_matrix_enc.dropna(axis=1, how='all', inplace=True)
            imputed = imp.fit_transform(f_matrix_enc.values)
            fm_cleaned = pd.DataFrame(imputed, columns=f_matrix_enc.columns)

            # fm_cleaned.columns = f_matrix_enc.columns
            output_df = fm_cleaned
            if self.export_features:
                path = FeaturesBuilder.get_definitions_file(self.client, target)
                ft.save_features(features_enc, path)
                logging.info("Exported features file: {0}".format(path))
        if output_df is not None and self.export_features:
            export_path = self.client_fs.get(['features', 'initial_build_feature_data_{0}.csv'.format(
                FeaturesBuilder.encode_name(target))])
            output_df.to_csv(export_path)
        return {
            'definitions': feature_definitions,
            'matrix': output_df
        }

    def aggregate_for_all_targets(self, df, fill_targets=False):
        '''

        :param df:
        :param fill_targets:
        :return: dict with target : data
        '''
        from input.consumers import DataConsumer
        output = {}
        target_names = [x['column'] for x in self._targets]
        dc = DataConsumer(target_names, [], grouping_function=self.grouping_function, save_to_db=False)
        dc.set_feature_function(self.feature_function)
        f_matrix = dc.calculate_df(df)
        for target in target_names:
            full_target = get_target_column_name(target)
            # Target is a feature, we already have it
            if target in f_matrix:
                f_matrix[full_target] = f_matrix[target]
            else:  # Target is present in raw data
                f_matrix[full_target] = df[target]
        for target in target_names:
            output[target] = f_matrix
        return output

    def get_features_wrapper(self, target, features):
        fwrapper = FeaturesWrapper(None, list(features[target].columns))
        return fwrapper

    def extract_for_all_targets(self, df, fill_target=False):
        """

        :param df:
        :param fill_target:
        :return:
        """
        output = dict()
        if fill_target:
            df = self._target_picker.fill_target_values(df)
        # Use the cache if we have it
        target_names = [x['column'] for x in self._targets]
        if self.has_cached_features():
            f_matrix = self.data_consumer.get_features(df, target_names)
            for t in target_names:
                full_target = get_target_column_name(t)
                if full_target not in df:
                    df[full_target] = 0
                f_matrix[full_target] = df[full_target]
            for t in target_names:
                output[t] = f_matrix
        else:
            for t in target_names:
                ftrs = self.extract_features(df, t, fill_target=False)
                output[t] = ftrs
        return output

    def __extract_with_fn(self, df, target, fn):
        """
        Extracts features using a feature generation function.
        :param df:
        :param target:
        :param fn:
        :return:
        """
        from input.consumers import flatten_feature_values
        dict_c = df.to_dict('records')
        # Run with grouping
        output_dicts = fn(dict_c)
        output_dicts = flatten_feature_values(output_dicts)
        output_df = pd.DataFrame([output_dicts])  # We flattened it, so we must convert it to an array of dicts
        return output_df

    def extract_features(self, df, target, fill_target=False):
        """

        :param df:
        :param target:
        :param fill_target:
        :return:
        """
        from featuretools import load_features
        start_time = time.time()
        target_fullname = get_target_column_name(target)
        if self.feature_function is not None:
            f_matrix = self.__extract_with_fn(df, target, self.feature_function)
        elif not self.use_feature_generation:
            # df.drop('timestamp', axis=1, inplace=True)
            df.dropna(axis=0, how='any')
            f_matrix = df
        else:
            # trained_with = __get_training_set()
            # df = trained_with.append(df, ignore_index=True)
            if target_fullname not in df:
                df[target_fullname] = 0
            tid_ = target_fullname + '_id'
            df[tid_] = range(len(df))
            df[target] = 0
            # Load features from the data consumer or from featuretools
            if not self.has_cached_features():
                raise Exception("No data consumer added.")
            f_matrix = self.data_consumer.get_single_target_features(df, target)
            # else:
            #    features = FeaturesBuilder.get_feature_definitions(target)
            #    es, feature_input = load_prediction_features(df, target_fullname, df_timestamp,
            #                                                 timestamp_field=self.timestamp_field)
            #    f_matrix = ft.calculate_feature_matrix(features, entityset=es)

        elapsed_time = time.time() - start_time
        logging.info("Feature extraction for `{0}` took ".format(target) + str(elapsed_time) + "")
        if self.use_feature_generation and not fill_target:
            if target_fullname in f_matrix:
                f_matrix = f_matrix.drop(target_fullname,
                                         axis=1)  # We remove the target feature, because the model was not trained with it..

        return f_matrix

    def load_dataset(self, df, entity_target, settings, cleanup_lambda=None, fetch_target=True):
        """
        :param df: 
        :param target: 
        :param settings: 
        :param cleanup_lambda: 
        :param fetch_target: 
        """
        if fetch_target:
            df = self._target_picker.fill_target_values(df)
        if cleanup_lambda is not None:
            df = cleanup_lambda(df)
        c = self._collection
        assert 'index' in settings
        assert 'timestamp' in settings
        index = settings['index']
        timestamp_column = settings['timestamp']

        target_value_column = get_target_column_name(entity_target)
        tid_ = target_value_column + '_id'
        logging.info("Loading dataset with target id: " + tid_)
        if tid_ not in df:
            df[tid_] = range(len(df))
        # Create the base entity set and noralize the target
        es = ft.EntitySet(self._dc.model_name)
        es = es.entity_from_dataframe(entity_id='base_entity',
                                      dataframe=df,
                                      index=index,
                                      time_index=timestamp_column)
        es = es.normalize_entity(base_entity_id='base_entity',
                                 new_entity_id=target_value_column,
                                 index=tid_,
                                 additional_variables=[target_value_column])
        # If we have any internal entities for the current collection, use them
        if self.internal_entities:
            internals = filter(lambda x: x.collection == c.key, self.internal_entities)
            for e in internals:
                es = es.normalize_entity(base_entity_id=c.name,
                                         new_entity_id=e.name,
                                         index=e.index,
                                         additional_variables=e.fields)
        for r in self._dc.relations:
            e1 = r[0].split('.')
            e2 = r[1].split('.')
            rl = ft.Relationship(es[e1[0]][e1[1]], es[e2[0]][e2[1]])
            es = es.add_relationship(rl)
        return es

    def __convert_to_donut__(self, fds):
        from code_generation import generate_features_module
        target_ftr_defs = dict()
        for full_target_name in fds:
            target_ftr_defs[full_target_name] = fds[full_target_name]['definitions']
        feature_dicts = format_feature_sets(target_ftr_defs, self.timestamp)
        features_module = generate_features_module(feature_dicts)
        if self.has_cached_features():
            self.data_consumer.set_feature_definitions(feature_dicts)
        script = feature_dicts
        return {
            'features': script,
            'module': features_module
        }

    @staticmethod
    def encode_name(v):
        """Encodes a feature definition file name
        """
        import hashlib
        h = hashlib.new('ripemd160')
        h.update(v.encode('utf-8'))
        ve = h.hexdigest()
        return ve

    @staticmethod
    def get_definitions_file(client, target):
        from helpers import ClientFs
        h_file = FeaturesBuilder.encode_name(target)
        client_fs = ClientFs(client)
        filename = "ftrdef_{0}.dill".format(h_file)
        path = client_fs.get(['features', filename])
        return path

    def _get_collection_data(self):
        reader = self.reader()  # MongoBufferedReader()
        c = self._collection
        fields = [cc.toDict() for cc in c.fields]
        df = reader.read_as_pandas_frame(c.key, c.start, c.end, fields=fields)
        internal_indexes = c.internal_entity_keys
        # Make sure we generate our normalization columns
        if internal_indexes != None and len(internal_indexes) > 0:
            for iindex in internal_indexes:
                if not iindex in df.columns:
                    df[iindex] = range(len(df))
        index = c.index if c.index != None else '_id'
        df.dropna(axis=0, how='any')
        reader.close()
        return df, index

    @staticmethod
    def variable_feature_to_dict(ftr, timestamp_column, root_feature=None):
        variable = ftr.variable if hasattr(ftr, 'variable') else None
        variable_index = None
        if variable is not None and variable.id.startswith('first_base_entity'):
            variable_index = 'first'
        if variable is not None and 'base_entity_time' in variable.id:
            id = variable = timestamp_column
            assert variable is not None
        else:
            id = variable.id
        output = {
            'name': ftr.name,
            'key': id,
            'index': variable_index,
            'type': 'direct'
        }
        return output

    @staticmethod
    def feature_to_dict(target, fdef, depth=0, timestamp_column='timestamp'):
        from featuretools.primitives.binary_transform import BinaryFeature
        base_feature = fdef.base_features[0] if len(fdef.base_features) > 0 else None
        output = dict()
        ftr_name = fdef.get_name()
        if fdef.name == 'num_unique':
            base_value = FeaturesBuilder.feature_to_dict(target, base_feature, depth=(depth + 1),
                                                         timestamp_column=timestamp_column)
            output = {
                'type': 'predicate',
                'key': base_value,
                'name': 'num_unique',
            }
        else:
            if isinstance(fdef, BinaryFeature):
                output = FeaturesBuilder.feature_to_dict(target, base_feature, depth=(depth + 1),
                                                         timestamp_column=timestamp_column)
                output = {
                    'left': output,
                    'right': fdef.right,
                    'op': fdef.operator,
                    'type': 'binary'
                }
            elif fdef.name == 'isin':
                output = FeaturesBuilder.feature_to_dict(target, base_feature, depth=(depth + 1),
                                                         timestamp_column=timestamp_column)
                output = {
                    'name': 'in',
                    'in_values': fdef.list_of_outputs,
                    'key': output,
                    'type': 'predicate'
                }
            else:
                if hasattr(fdef, 'variable'):
                    tmpout = FeaturesBuilder.variable_feature_to_dict(fdef, timestamp_column)
                if base_feature is None:
                    output = tmpout
                else:
                    base_out = FeaturesBuilder.feature_to_dict(target, base_feature, depth=(depth + 1),
                                                               timestamp_column=timestamp_column)
                    fname = fdef.name
                    if fname == 'not':
                        fname = 'isnot'
                    output = {
                        'name': fname,  # the op to eval
                        'key': base_out,  # the key for the op from the data
                        'index': None,
                        'type': 'direct'
                    }
        output['title'] = ftr_name
        return output

    def has_cached_features(self):
        return self.data_consumer is not None

    def _format_features_gen_input(self, df, fields_formatting=None):
        if fields_formatting is not None:
            for field in fields_formatting:
                if field not in df:
                    continue
                f_type = fields_formatting[field]
                df[field] = df[field].astype(f_type)
        return df

    def generate_features(self, fields_formatting=None):
        """Creates features definitions for each target
        :type fields_formatting: dict Formatting to perform to fields { fld: type }
        """
        try:
            df, index = self._get_collection_data()
            timestamp = self.timestamp if 'timestamp' in self._collection else None
            df.dropna(axis=0, how='any')
            targets = self._targets
            target_columns = [t.column for t in targets]
            df = create_target_columns(df, target_columns)
            df = self._format_features_gen_input(df, fields_formatting)
            feature_sets = dict()
            for target in target_columns:
                full_target_name = get_target_column_name(target)
                df_without_other_targets = df.copy()
                # Remove other targets
                other_targets = [get_target_column_name(t) for t in target_columns if
                                 get_target_column_name(t) != full_target_name]

                df_without_other_targets = df_without_other_targets.drop(other_targets, axis=1)
                created_features = self.create(df_without_other_targets, target, df_settings={
                    'index': index,
                    'timestamp': timestamp
                })
                definitions, ftrs = (created_features['definitions'], created_features['matrix'])
                feature_sets[full_target_name] = {
                    'definitions': definitions,
                    'features': ftrs,
                    'orig_target': target
                }
                print("Created features for {0}".format(target))
            self.feature_definitions = feature_sets
            donut = self.__convert_to_donut__(feature_sets)
            return donut
        except  Exception as e:
            log.error(str(e))
            log.error(traceback.format_exc())


class FeaturesWrapper(object):
    def __init__(self, collection_name=None, features=None):
        self.features = dict()
        self.collections = []
        if collection_name is not None:
            self.add_collection(collection_name, add_features=features is None)
        if features is not None:
            self.features = dict()
            for f in features:
                self.features[f] = {
                    'name': f
                }

    def get_feature_avg_aggregates(self, selector_prefix):
        output = dict()
        for feature in self.features:
            if feature['type'] != 0: continue
            output[feature['name']] = {"$avg": "{0}{1}".format(selector_prefix, feature['name'])}
        return output

    def add(self, name):
        self.features[name] = {
            'name': name
        }

    def add_collection(self, collection_name, add_features=True):
        if not collection_name in self.collections:
            if add_features:
                collection = settings.get_db()[collection_name]
                document = collection.find_one({})
                if not document is None:
                    for key in document:
                        if (key == "_id"):
                            continue
                        self.features[key] = {"name": key}
            self.collections.append(collection_name)

    def get_values(self, dict_values):
        """Gets feature values"""
        outp = []
        for f in self.features:
            val = dict_values.get(f['name'])
            if val is None: val = 0
            outp.append(val)
        return outp

    def get_names(self):
        return [self.features[f]['name'] for f in self.features]

    def pair_with_importance(self, importances):
        i = 0
        output = []
        if isinstance(importances, int):
            for f in self.features:
                output.append(
                    {'name': self.features[f]['name'], 'importance': 0})
                i += 1
        else:
            importances = np.ravel(importances)
            for f in self.features:
                output.append(
                    {'name': self.features[f]['name'],
                     'importance': importances[i] if (i) < len(importances) else 'none'})
                i += 1

        return output

    def drop(self, feature_names):
        for f in [x for x in feature_names if x in self.features]:
            del self.features[f]

    @staticmethod
    def create(collection_name, features=None):
        fw = FeaturesWrapper(collection_name, features)
        return fw

    @staticmethod
    def from_data(df):
        fw = FeaturesWrapper(None, list(df.columns))
        return fw
