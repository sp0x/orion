from unittest import TestCase
import sys
from ml.nlp import *
from ..test_helpers import generate_language_df


class TestNlp(TestCase):

    def test_analyze_column_language(self):
        test_df = generate_language_df('test_column', rows=100)
        # test_df.to_csv('testdf.csv')
        language_stats = analyze_column_language(test_df, 'test_column')
        assert language_stats['en'] == 40
        assert language_stats['bg'] == 36
        assert language_stats['mk'] == 3
        assert language_stats['de'] == 20
        assert language_stats['INV'] == 1
        assert language_stats['by_cols']['test_column']['en'] == 40
        assert language_stats['by_cols']['test_column']['bg'] == 36
        assert language_stats['by_cols']['test_column']['de'] == 20
        assert language_stats['by_cols']['test_column']['INV'] == 1
        assert 'test_column.lng' in test_df
        invalids = test_df[test_df['test_column.lng'] == 'INV']
        assert len(invalids) == 1

    def test_analyze_column_language_cnn(self):
        test_df = generate_language_df('test_column', rows=100)
        # test_df.to_csv('testdf.csv')
        language_stats = analyze_column_language_batch(test_df, 'test_column', langdetect_cnn_batch)
        assert language_stats['en'] == 40
        assert language_stats['bg'] == 36
        assert language_stats['uk'] == 3
        assert language_stats['de'] == 19
        assert language_stats['nl'] == 1
        assert language_stats['by_cols']['test_column']['en'] == 40
        assert language_stats['by_cols']['test_column']['bg'] == 36
        assert language_stats['by_cols']['test_column']['de'] == 19
        assert language_stats['by_cols']['test_column']['nl'] == 1
        assert 'test_column.lng' in test_df
        bgs = test_df[test_df['test_column.lng'] == 'bg']
        assert len(bgs) == 36
    # def test_traincnn(self):
    #     from ml.nlp.cnndetector import train
    #     import os
    #     cr_dir = os.path.abspath(os.path.dirname(__file__))
    #     data_dir = os.path.join(cr_dir, '..', '..', 'ml', 'nlp', 'data', 'ted500')
    #     train(data_dir=data_dir, num_epoch=1)

    # def test_cnn_prep(self):
    #     import os
    #     import glob
    #     from ml.nlp.utils import TextReader
    #     cr_dir = os.path.abspath(os.path.dirname(__file__))
    #     data_dir = os.path.join(cr_dir, '..', '..', 'ml', 'nlp', 'data', 'ted500')
    #     class_names = [c.split('.')[-1] for c in glob.glob(os.path.join(data_dir,'ted_500.*'))]
    #     reader = TextReader(data_dir, class_names=class_names)
    #     reader.prepare_data(vocab_size=4090, test_size=50)

    #
    # def test_benchmark_cnn(self):
    #     from ml.nlp.cnndetector import predict_batch
    #     from timeit import default_timer as timer
    #     import pandas as pd
    #     df = generate_language_df('test_column', 100)
    #     times = []
    #     t = 1
    #     for i in range(t):
    #         start = timer()
    #         language_stats = analyze_column_language_batch(df, 'test_column', predict_batch)
    #         end = timer()
    #         times.append(end - start)
    #     t2 = pd.Series(times).mean()
    #     print("\n CNN Elapsed: " + str(t2))  # Time in seconds

    # def test_benchmark(self):
    #     from timeit import default_timer as timer
    #     import pandas as pd
    #     df = generate_language_df('test_column', 100)
    #     times = []
    #     t = 1
    #     # for i in range(t):
    #     #     start = timer()
    #     #     language_stats = analyze_column_language(df, 'test_column')
    #     #     end = timer()
    #     #     times.append(end - start)
    #     # t1 = pd.Series(times).mean()
    #     # times = []
    #     t1 = 1
    #     #CNN
    #     for i in range(t):
    #         start = timer()
    #         language_stats = analyze_column_language(df, 'test_column',langdetect_cnn)
    #         end = timer()
    #         times.append(end - start)
    #
    #     t2 = pd.Series(times).mean()
    #
    #     print("\n PyLangdetect Elapsed: " + str(t1))  # Time in seconds
    #     print("\n CNN Elapsed: " + str(t2))  # Time in seconds
