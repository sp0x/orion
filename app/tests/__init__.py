import unittest
import os
import sys

# from neural_network_tests import NeuralNetworkTest
# from ml import neural_networks
# from utils import load_json, par_path
# from constants import KERAS_CLASS_PATH
# neural_networks.layer_list = load_json(par_path(KERAS_CLASS_PATH))
# suite = unittest.TestLoader().loadTestsFromTestCase(NeuralNetworkTest)
# unittest.TextTestRunner().run(suite)
# from feature_extractor_test import FeatureExtractorTest
# So that we can run this with python via command line
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
#sys.path.append(os.getcwd())

from .ml_tests.test_storage import TestMlStorage
from .ml_tests.test_nlp import TestNlp
# loader = unittest.TestLoader()
# tests = [
#     loader.loadTestsFromTestCase(test) for test in (
#         NlpTests, TestMlStorage
#     )
# ]
# suite = unittest.TestSuite(tests)
# runner = unittest.TextTestRunner(verbosity=2)
# runner.run(suite)

#
# from tests.test_targetPicker import TestTargetPicker
# from tests.test_featuresBuilder import TestFeaturesBuilder
# from tests.test_dataframeParser import TestDataframeParser
# from tests.ml_tests.storage import TestMlStorage
# # suite = unittest.TestLoader().loadTestsFromTestCase(OneClickE2ETest)
#
# #suite = unittest.TestLoader().loadTestsFromTestCase(TestTargetPicker)
# #result = unittest.TextTestRunner().run(suite)
#
#
# #suite = unittest.TestLoader().loadTestsFromTestCase(TestFeaturesBuilder)
# #suite = unittest.TestLoader().loadTestsFromTestCase(TestDataframeParser)
# #suite = unittest.TestLoader().loadTestsFromTestCase(TestFeaturesBuilder)
# suite = unittest.TestLoader().loadTestsFromTestCase(TestMlStorage)
# result = unittest.TextTestRunner().run(suite)
#unittest.main()
