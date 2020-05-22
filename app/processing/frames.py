import pandas as pd
import io
import boto3
import numpy as np

from ml.nlp import analyze_column_language_batch
from stat_utils import hist, freq
from utils import resolve_type

_frame_handlers = dict()


def get_summary(df, fields_formatting=None):
    dfp = DataframeParser(df)
    return dfp.get_summary(fields_formatting)


def frame_src_type(tp):
    def decorator(cls):
        global _frame_handlers
        _frame_handlers[tp] = cls
        return cls
    return decorator


def get_frame_parser(tp, *args, **kwargs):
    """

    :param tp: The type of parser that you need
    :param args: args for the parser
    :param kwargs:
    :return:
    """
    global _frame_handlers
    cls = _frame_handlers.get(tp, None)
    return cls(*args, **kwargs) if cls is not None else None


def enrich_text_columns(df, columns):
    """
    Add info about text columns
    :param df:
    :param columns:
    :return:
    """
    stats = {'lang': analyze_column_language_batch(df, columns)}
    return stats

@frame_src_type("pandas")
class DataframeParser(object):
    def __init__(self, df=None):
        self.df = df

    def setdf(self, df):
        self.df = df


    def get_summary(self, fields_formatting=None, do_language_detection=True):
        """

        :param do_language_detection:
        :param fields_formatting:
        :return:
        """
        dtypes = dict(self.df.dtypes)
        scheme = {k: resolve_type(v, fields_formatting[k] if (fields_formatting is not None and
                                                              k in fields_formatting) else None)
                  for k, v in dtypes.items()}
        desc = {}
        text_cols = []
        for col in self.df:
            from sklearn.metrics.classification import type_of_target
            tmp = pd.DataFrame()
            tmp[col] = self.df[col]
            col_data = tmp.as_matrix().ravel()
            target_type = type_of_target(col_data)
            col_meta = {'target_type': target_type}
            if scheme[col] == 'str' and target_type!='binary':
                text_cols.append(col)
            if scheme[col] == 'bool' or scheme[col] == 'object' or scheme[col] == 'str':
                desc[col] = freq(self.df, col)
            elif 'datetime' in scheme[col]:
                desc[col] = {'type': 'range',
                             'values': {'min': str(self.df[col].min()), 'max': str(self.df[col].max())}}
            else:
                desc[col] = hist(self.df, col, 'fd')
            for meta_prop in col_meta:
                desc[col][meta_prop] = col_meta[meta_prop]
        if do_language_detection:
            text_stats = enrich_text_columns(self.df, text_cols)
            lang_stats = text_stats['lang']
            columns_lang_stats = lang_stats['by_cols']
            for txt_col in lang_stats['by_cols']:
                desc[txt_col]['languages'] = columns_lang_stats[txt_col]
        else:
            for txt_col in text_cols:
                desc[txt_col]['languages'] = {}
        return {
            'scheme': scheme,
            'desc': desc
        }
