from datetime import datetime, timedelta
import time
import settings
import logging
import pandas as pd
import numpy as np
import pydevd
from db.data_handlers import MongoBufferedReader
from db.encoding import EncodingHelper
from ml import DataException, MlScript, load_model
from ml.feature_extraction import FeaturesBuilder

logging.basicConfig(level=logging.DEBUG)

accuracy_alert_threshold = 400
accuracy_alert_threshold2 = 1600


def scale(mod_details, df, data_type):
    """

    :param mod_details: dict Model details
    :param df: pandas.DataFrame a dataframe to be scaled
    :param data_type: string Either `data` or `targets`
    :return:
    """
    if 'scaling' not in mod_details:
        return df.as_matrix()
    elif data_type == 'data' and 'data' not in mod_details['scaling']:
        return np.nan_to_num(df.as_matrix())
    elif data_type == 'data' and 'data' in mod_details['scaling'] and mod_details['scaling']['data'] is not None:
        return mod_details['scaling']['data'].transform(np.nan_to_num(df.as_matrix()))
    elif data_type == 'data' and 'data' in mod_details['scaling'] and mod_details['scaling']['data'] is None:
        return np.nan_to_num(df.as_matrix())

    elif data_type == 'targets' and 'targets' not in mod_details['scaling']:
        return np.nan_to_num(df.as_matrix())
    elif data_type == 'targets' and 'targets' in mod_details['scaling'] and mod_details['scaling'][
        'targets'] is not None:
        return mod_details['scaling']['targets'].transform(np.nan_to_num(df.as_matrix()))
    elif data_type == 'targets' and 'targets' in mod_details['scaling'] and mod_details['scaling']['targets'] is None:
        return np.nan_to_num(df.as_matrix())


def scaleback(mod_details, ndarr, data_type):
    if 'scaling' not in mod_details:
        return ndarr
    elif data_type == 'data' and 'data' not in mod_details['scaling']:
        return np.nan_to_num(ndarr)
    elif data_type == 'data' and 'data' in mod_details['scaling'] and mod_details['scaling']['data'] is not None:
        return mod_details['scaling']['data'].inverse_transform(np.nan_to_num(ndarr))
    elif data_type == 'data' and 'data' in mod_details['scaling'] and mod_details['scaling']['data'] is None:
        return np.nan_to_num(ndarr)

    elif data_type == 'targets' and 'targets' not in mod_details['scaling']:
        return np.nan_to_num(ndarr)
    elif data_type == 'targets' and 'targets' in mod_details['scaling'] and mod_details['scaling'][
        'targets'] is not None:
        return mod_details['scaling']['targets'].inverse_transform(np.nan_to_num(ndarr))
    elif data_type == 'targets' and 'targets' in mod_details['scaling'] and mod_details['scaling']['targets'] is None:
        return np.nan_to_num(ndarr)


class PredictionHelper:
    def __init__(self,
                 script: MlScript,
                 source_collection,
                 targets_definitions,
                 timestamp_column=None,
                 use_feature_generation=False,
                 index_column=None):
        self.script = script
        self.prediction_log = []
        self.db = settings.get_db()
        self.predictions = self.db['predictions']
        self.client = script.client
        # target_columns, timestamp_column
        fbuilder_config = {
            'db': self.db,
            'collection': {
                'key': source_collection
            },
            'targets': targets_definitions,
            'use_feature_generation': use_feature_generation
        }
        self.features_builder = FeaturesBuilder(fbuilder_config, MongoBufferedReader)
        self.features_builder.set_script(script)
        self.targets = [x['column'] for x in targets_definitions]
        self.timestamp_column = timestamp_column
        self.data_consumer = None
        self.feature_generation_enabled = True
        self.targets_models = dict()

    @staticmethod
    def from_script(script: MlScript):
        flags = script.input_flags()
        collection = flags['collection']['key']
        timestamp_column = flags['collection']['timestamp']
        index_column = flags['collection']['index']
        targets = script.get_targets()
        ph = PredictionHelper(script, source_collection=collection,
                              targets_definitions=targets,
                              timestamp_column=timestamp_column, index_column=index_column,
                              use_feature_generation=script.use_featuregen)
        return ph

    def set_feature_generation(self, enabled):
        self.feature_generation_enabled = enabled
        # self.f_builder.enable_feature_generation = enabled

    def set_data_consumer(self, consumer):
        self.data_consumer = consumer
        # Set a data consumer to the feature builder, so it can use it to look up features.
        # self.f_builder.set_data_consumer(consumer)

    def accuracy_alert(self, info):
        from slackclient import SlackClient
        import os
        slack_token = os.environ["SLACK_API_TOKEN"]
        sc = SlackClient(slack_token)
        r = sc.api_call(
            "chat.postMessage",
            channel="#prediction_alerts",
            text="Prediction score alert: \n```" + str(info) + '```',
            user='intelflows'
        )

    def get_performance(self, cr_df):
        import sklearn
        cr_ts = cr_df['timestamp'][0]
        cr_target = cr_df['pm10'][0]
        hour = timedelta(hours=1)
        hour2 = timedelta(hours=2)
        for p in reversed(self.prediction_log):
            pts = p['ts']
            pval = p['value']
            time_diff = cr_ts - pts
            if hour < time_diff < hour2:
                v1 = [pval]
                v2 = [cr_target]
                accuracy = sklearn.metrics.mean_squared_error(v1, v2)
                logging.info("Performance: " + str(accuracy))
                if accuracy > accuracy_alert_threshold:
                    self.accuracy_alert({'score': accuracy, 'predicted': pval, 'actual': cr_target})
                    break

    def encode_df(self, df):
        """
        Encodes the fields in the desired way, depending on the script  that was passed in.
        :param df:
        :return:
        """
        flags = self.script.input_flags()
        encoding_helper = EncodingHelper(flags)
        for f in flags['fields']:
            if 'encoding' not in f:
                continue
            df = encoding_helper.fill_encoded(df, f)
        return df

    def predict_raw(self, df, padding_rows, use_cache=False):
        df_without_cache = None
        if use_cache:
            prediction_cache = self.get_prediction_cache(df, inplace=True)
            df_without_cache = df.copy()
            for t in self.targets:
                d = df_without_cache[df_without_cache['prediction_value_{0}'.format(t)] == -1]
                logging.info("items_without prediction cache for {0}".format(t) + str(len(d)))
                df_without_cache = d

        if df_without_cache is not None and len(df_without_cache) == 0:
            predictions = df[['prediction_value_{0}'.format(t) for t in self.targets]]
            output = []
            for ix, prow in predictions.iterrows():
                p_dict = dict()
                for _t, pvalue in [(t, prow['prediction_value_{0}'.format(t)]) for t in self.targets]:
                    p_dict[_t] = float(pvalue)
                output.append(p_dict)
            return output
        else:
            if use_cache:
                non_target_df = df.drop(['prediction_value_{0}'.format(t) for t in self.targets], axis=1)
            else:
                non_target_df = df
            # A dict with `target` -> feature values
            target_feature_dict = self.features_builder.extract_for_all_targets(non_target_df)  # target_value
            first_df = target_feature_dict[list(target_feature_dict.keys())[0]]
            # row_diff = len(df) - len(df_without_cache)
            # logging.info(row_diff)
            result = self.predict_target_dict(target_feature_dict, first_df, padding_rows)  # nontargetdf
            return result

    def df_matches_model(self, df, target):
        model_details = self.get_model_details(target)
        model_cols = model_details['columns']
        m_target = model_cols['target']
        if target != m_target:
            return False, []
        missing_fields = []
        # Check for missing features
        for ml_col in model_cols['data']:
            if ml_col not in df:
                missing_fields.append(ml_col)
        return len(missing_fields) == 0, missing_fields

    def target_is_encoded(self, target_name):
        fields = self.script.input_flags()['fields']
        for field in fields:
            if field['name'] == target_name:
                encoding = field['encoding'] if 'encoding' in field else None
                return encoding is not None and len(encoding) > 0

    def predict_target_dict(self, features, df, padding_rows):
        """

        :param features:
        :param df:
        :param padding_rows:
        :return:
        """
        import ml.cleanup
        start_time = time.time()
        prediction_output = []
        predictions_targets = dict()
        cnt_predictions = 0
        # logging.info(df.shape)
        # logging.info(features[list(features.keys())[0]].shape)
        for target in self.targets:
            model_details = self.get_model_details(target)
            ml_model = model_details['model']
            # Validate our input
            df_is_valid, missing_fields = self.df_matches_model(df, target)
            if not df_is_valid:
                raise DataException("Your df\'s features do not match the one the model was built with. Check your "
                                    "input.\n "
                                    " Missing fields: " + str(missing_fields))
            features_to_predict_on = features[target]
            features_to_predict_on = features_to_predict_on.iloc[
                                     (-1 * padding_rows):]  # Get the last rows, because we'll predict on it.
            cnt_predictions = features_to_predict_on.shape[0]
            if not self.feature_generation_enabled:
                cleaned_df = ml.cleanup.clean_data_for_prediction(features_to_predict_on)
            else:
                cleaned_df = features_to_predict_on
            # TODO: FIX this, bug in multiple targets
            cleaned_df = ml.cleanup.clean_data_for_prediction(cleaned_df)
            cleaned_df = ml.cleanup.drop_targets(self.targets, cleaned_df)

            # cleaned_df.to_csv('pred_input_{0}.csv'.format(target))
            pred_input = scale(model_details, cleaned_df, 'data')
            prediction = ml_model.predict(pred_input)
            prediction = prediction.reshape(-1, 1)
            prediction = scaleback(model_details, prediction, 'targets')
            predictions_targets[target] = prediction.tolist()
        for p in range(cnt_predictions):
            output_entry = dict()
            for t in self.targets:
                output_entry[t] = predictions_targets[t][p]
            prediction_output.append(output_entry)
        elapsed_time = time.time() - start_time
        logging.info("Prediction took " + str(elapsed_time) + "")
        # Validate and log if necessary
        ix = 0
        for index, predicted_on_row in df.iterrows():
            try:
                predictions = dict()
                encoding_helper = EncodingHelper(self.script.input_flags())
                for t in self.targets:
                    prediction = predictions_targets[t][ix]
                    # TODO: Make this work only for classifications
                    if self.target_is_encoded(t):
                        for i, pred_value in enumerate(prediction):
                            plaintext_value = encoding_helper.decode_plaintext(pred_value, t)
                            prediction[i] = plaintext_value

                    predictions[t] = prediction
                self.__store_prediction(predicted_on_row, predictions)
            except:
                pass

            ix = ix + 1
            # self.prediction_log.append({'ts': df['timestamp'][0], 'value': prediction[0]})

        return prediction_output

    def predict(self, target, features, df, padding_rows):
        start_time = time.time()
        features_to_predict_on = features.iloc[(-1 * padding_rows):]  # Get the last row, because we'll predict on it.
        model = self.get_model_details(target)
        if not self.feature_generation_enabled:
            # cleaned_df = features_to_predict_on.drop('timestamp', axis=1)
            cleaned_df = features_to_predict_on.drop('timestamp', axis=1)
            # cleaned_df = cleaned_df.drop('pm10', axis=1) # we're using the current pm10
        else:
            cleaned_df = features_to_predict_on
        prediction = model.predict(cleaned_df).tolist()
        prediction_output = []
        for p in prediction:
            prediction_output.append({
                target: p
            })
        elapsed_time = time.time() - start_time
        logging.info("Prediction took " + str(elapsed_time) + "")
        # Validate and log if necessary
        ix = 0
        for index, predict_on_row in df.iterrows():
            self.__store_prediction(predict_on_row, prediction[ix])
            ix = ix + 1
        # self.prediction_log.append({'ts': df['timestamp'][0], 'value': prediction[0]})

        return prediction_output

    def get_prediction_cache(self, df, inplace=False):
        output = dict()
        i = 0
        for ix, row in df.iterrows():
            ts = pd.to_datetime(row['timestamp']).to_pydatetime()
            query = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'timestamp': ts
            }
            p = self.predictions.find_one(query)
            if p is not None and inplace:
                for t in self.targets:
                    value__t = 'prediction_value_' + t
                    if value__t not in df:
                        df[value__t] = -1.0
                    df[value__t][ix] = float(p['predicted'][t])
            output[i] = p['predicted'] if p is not None else None
            i += 1
        return output

    def __store_prediction(self, for_data, output):
        ts = pd.to_datetime(for_data['timestamp']).to_pydatetime()
        query = {
            'latitude': for_data['latitude'],
            'longitude': for_data['longitude'],
            'timestamp': ts
        }
        record = {
            'latitude': for_data['latitude'],
            'longitude': for_data['longitude'],
            'timestamp': ts,
            'predicted': output
        }
        self.predictions.update(query, {"$set": record}, upsert=True)
        return record

    def get_model_details(self, target):
        from ml import load_model
        from ml.classifiers import Experiment
        if target in self.targets_models and self.targets_models[target] is not None:
            return self.targets_models[target]
        else:
            mlmodel = load_model(self.client, self.script.model_id, target)
            self.targets_models[target] = mlmodel
        return self.targets_models[target]

    def set_model(self, target, model):
        self.targets_models[target] = model
