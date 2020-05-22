from unittest import TestCase
import pandas as pd
from processing import get_frame_parser
from tests.test_helpers import get_testfile
from datetime import datetime


class TestDataframeParser(TestCase):

    def test_get_summary(self):
        df = pd.read_csv(get_testfile('intelflows.cities.csv'))
        df = df.head(1000)
        dfp = get_frame_parser("pandas", df)
        summary = dfp.get_summary({'device_ID': str, 'timestamp': datetime}, do_language_detection=False)
        self.assertTrue('scheme' in summary)
        self.assertTrue('desc' in summary)
        self.assertTrue(summary['scheme']['device_ID'] == 'str')
        self.assertTrue(summary['scheme']['timestamp'] == 'datetime')


    def test_get_summary_file(self):
        df_file = get_testfile('intelflows.cities.csv')
        dfp = get_frame_parser("file", df_file)
        summary = dfp.get_summary({'device_ID': str, 'timestamp': datetime}, do_language_detection=False)
        self.assertTrue('scheme' in summary)
        self.assertTrue('desc' in summary)
        self.assertTrue(summary['scheme']['device_ID'] == 'str')
        self.assertTrue(summary['scheme']['timestamp'] == 'datetime')