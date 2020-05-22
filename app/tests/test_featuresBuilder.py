from unittest import TestCase
import mongomock

from input.consumers import DataConsumer
from tests.test_helpers import timeify, MockMongoBufferedReader
import pandas as pd
import mock
import json
from datetime import datetime
from processing.frames import get_summary


class TestFeaturesBuilder(TestCase):

    def setUp(self):
        from tests.test_helpers import fill_db
        db = mongomock.MongoClient().db
        self.key = 'collectionIntelflows'
        self.key_features = 'iflowsFeatures'
        self.collection = fill_db(db, self.key, 'intelflows.cities.csv', lambda x: timeify(x, 'timestamp'), cnt=100)
        self.ftrs_collection = db[self.key_features]
        self.fields_formatting = {'device_ID': str, 'timestamp': datetime}
        self.data_config = {
            'db': db,
            'model_name': 'model1',
            'client': 'clientExample1',
            'collection': {
                'name': 'device',
                'key': self.key,
                'start': '1/1/1970',
                'end': '1/1/1970',
                'index': '_id',
                'timestamp': 'timestamp'
            },
            'relations': [],
            'targets': [
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
                        {'key': 'city_id'},
                        {'key': 'device_ID'},
                    ]
                }
            ],
            'internal_entities': [{
                'collection': 'key',
                'name': 'pm10',
                'index': 'pm_id',
                'fields': ['_id', 'pm10']
            }],
            'use_feature_generation': True,
            'verbose': True,
            'export_features': True
        }

    @mock.patch('db.data_handlers.MongoBufferedReader')
    def test_generate_feature(self, mock_reader):
        mock_reader.return_value = MockMongoBufferedReader()
        from ml.feature_extraction import FeaturesBuilder
        fe = FeaturesBuilder(self.data_config, mock_reader)
        features = fe.generate_features(fields_formatting=self.fields_formatting)
        self.assertTrue('features' in features)
        self.assertTrue('module' in features)
        self.assertTrue(len(features['features']) > 0)
        self.assertTrue(len(features['module']) > 0)

    @mock.patch('db.data_handlers.MongoBufferedReader')
    def test_custom_feature(self, mockReader):
        mockReader.return_value = MockMongoBufferedReader(filepath='../data/test_fixtures/intelflows.cities.csv',
                                                          limit=100)
        from ml.feature_extraction import FeaturesBuilder
        f_builder = FeaturesBuilder(self.data_config, mockReader)
        dc = DataConsumer([
            'pm10'
        ], None,
            'number', db_settings={
                'data_collection': self.collection,
                'features_collection': self.ftrs_collection
            })
        f_builder.set_data_consumer(dc)
        # Code generation: Create a features file which defines functions for each feature
        # Make it so a feature can use the value of other features
        # Feature private variables

    @mock.patch('db.data_handlers.MongoBufferedReader')
    def test_extract_features(self, mockReader):
        mockReader.return_value = MockMongoBufferedReader(filepath='../data/test_fixtures/intelflows.cities.csv',
                                                          limit=100)
        from ml.feature_extraction import FeaturesBuilder
        f_builder = FeaturesBuilder(self.data_config, mockReader)
        dc = DataConsumer([
            'pm10'
        ], None,
            'number', db_settings={
                'data_collection': self.collection,
                'features_collection': self.ftrs_collection
            })
        f_builder.set_data_consumer(dc)
        donut = f_builder.generate_features(fields_formatting=self.fields_formatting)
        assert donut is not None
        record = self.collection.find_one({}, sort=[('number', -1)])
        df = pd.DataFrame([record])
        features = f_builder.extract_for_all_targets(df)
        assert features is not None
        assert 'pm10' in features
        self.assertIsInstance(features['pm10'], pd.DataFrame)
        assert len(features['pm10']) > 0
