import pandas as pd
import numpy as np


def freq(df, col, topn=5, other_label='auto'):
    s = (df[col].value_counts() / len(df[col])).round(4)
    res = {}
    if s.size > topn:
        other_label = other_label if other_label != 'auto' else 'other ({})'.format(s.size-4)
        res[other_label] = 1.0-s[:4].sum()
        res.update({str(k): v for k, v in dict(s[:4]).items()})
    else:
        res.update({str(k): v for k, v in dict(s).items()})
    return res

def hist(df, col, bins='auto', fallback_label='auto', fallback_bins=5):
    try:
        y, x = np.histogram(df[col].values, bins=bins, density=True)
        desc = {'type': 'hist', 'values': list(zip(x, y))}
    except:
        desc = {'type': 'freq', 'values': freq(df, col, fallback_bins, fallback_label)}
    return desc