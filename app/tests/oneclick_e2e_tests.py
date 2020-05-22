import unittest
import pandas as pd
import mock
from tests.test_helpers import MockMongoBufferedReader
import sys
print(sys.path)
from ml.classifiers import build, train


class MockFeaturesWrapper(object):
    def __init__(self):
        pass

    def add_collection(self, collection_name):
        pass

    def drop(self, feature_names):
        pass

class OneClickE2ETest(unittest.TestCase):

    # @mock.patch('db.data_handlers.MongoBufferedReader')
    # @mock.patch('ml.features.FeaturesWrapper.__init__')
    # @mock.patch('ml.features.FeaturesWrapper.add_collection')
    # @mock.patch('ml.features.FeaturesWrapper.drop')
    # def test_feature_gen_with_regression(self,drop, add, fw, mockReader):
    #     reader = MockMongoBufferedReader('/home/tony/Desktop/orion/app/data/test_fixtures/dengue-dataset.csv',['_id','case_id'])
    #     fw.return_value = None
    #     drop.return_value = None
    #     add.return_value = None
    #     mockReader.return_value = reader
    #     data_config = {
    #         'model_name': 'dengue',
    #         'collections': [{
    #             'name': 'dengue',
    #             'key': 'key',
    #             'start': '1/1/1970',
    #             'end': '1/1/1970',
    #             'index': '_id'
    #         }],
    #         'relations': [],
    #         'target': 'cases',
    #         'internal_entities': [{
    #             'collection': 'key',
    #             'name': 'cases',
    #             'index': 'case_id',
    #             'fields': ['_id', 'cases']
    #         }]
    #     }
    #     # from app.ml.feature_extraction import FeatureExtractor
    #     # fe = FeatureExtractor(data_config)
    #     # fe.generate_features()
    #     # reader.frame = fe._fm
    #     params = {'models':{'auto': {}},
    #               'options': {
    #                   'db': 'dengue',
    #                   'start': '1/1/1970',
    #                   'end': '1/1/1970',
    #                   'fields': [],
    #                   'scoring': 'neg_mean_squared_error'
    #               },
    #               'client': 'blah',
    #               'target': ['cases'],
    #               'task':'regression'
    #               }
    #     e = build(params)
    #     train(e)


    @mock.patch('db.data_handlers.MongoBufferedReader')
    @mock.patch('ml.features.FeaturesWrapper.__init__')
    @mock.patch('ml.features.FeaturesWrapper.add_collection')
    @mock.patch('ml.features.FeaturesWrapper.drop')
    def test_feature_gen_with_classification(self, drop, add, fw, mockReader):
        reader = MockMongoBufferedReader('/home/tony/Desktop/orion/app/data/test_fixtures/UFOPOLLUTANTS.csv',['_id','ET_id'])
        mockReader.return_value = reader
        fw.return_value = None
        drop.return_value = None
        add.return_value = None
        data_config = {
            'model_name': 'ufo',
            'collections': [{
                'name': 'ufo',
                'key': 'key',
                'start': '1/1/1970',
                'end': '1/1/1970',
                'index': '_id'
            }],
            'relations': [],
            'target': 'ET',
            'internal_entities': [{
                'collection': 'key',
                'name': 'ET',
                'index': 'ET_id',
                'fields': ['_id', 'ET']
            }]
        }
        # from app.ml.feature_extraction import FeatureExtractor
        # fe = FeatureExtractor(data_config)
        # fe.generate_features()
        # reader.frame = fe._fm
        params = {'models':{'auto': {}},
                  'options': {
                      'db': 'UFO',
                      'start': '1/1/1970',
                      'end': '1/1/1970',
                      'fields': [],
                      'scoring': 'accuracy'
                  },
                  'client': 'blah',
                  'target': ['ET']
                  }
        e = build(params)
        train(e)

    # @mock.patch('app.db.data_handlers.MongoBufferedReader')
    # @mock.patch('app.ml.features.FeaturesWrapper.__init__')
    # @mock.patch('app.ml.features.FeaturesWrapper.add_collection')
    # @mock.patch('app.ml.features.FeaturesWrapper.drop')
    # def test_feature_gen_with_high_dim_data(self, drop, add, fw, mockReader):
    #     reader = MockMongoBufferedReader('/home/tony/Desktop/orion/app/data/test_fixtures/Melbourne_housing_FULL.csv',['_id', 'Price_id'])
    #     fw.return_value = None
    #     drop.return_value = None
    #     add.return_value = None
    #     mockReader.return_value = reader
    #     data_config = {
    #         'model_name': 'mel',
    #         'collections': [{
    #             'name': 'mel',
    #             'key': 'key',
    #             'start': '1/1/1970',
    #             'end': '1/1/1970',
    #             'index': '_id'
    #         }],
    #         'relations': [],
    #         'target': 'Price',
    #         'internal_entities': [{
    #             'collection': 'key',
    #             'name': 'Price',
    #             'index': 'Price_id',
    #             'fields': ['_id', 'Price']
    #         }]
    #     }
    #     # from app.ml.feature_extraction import FeatureExtractor
    #     # fe = FeatureExtractor(data_config)
    #     # fe.generate_features()
    #     # reader.frame = fe._fm
    #     params = {'models':{'auto': {}},
    #                   'options': {
    #                       'db': 'mel',
    #                       'start': '1/1/1970',
    #                       'end': '1/1/1970',
    #                       'fields': [],
    #                       'scoring': 'neg_mean_squared_error'
    #                   },
    #                   'client': 'blah',
    #                   'target': ['Price'],
    #             'task':'regression'}
    #     e = build(params)
    #     train(e)