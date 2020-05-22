import pydevd
import pandas as pd
import numpy as np
import time

def convert_date_to_stamp(date):
    ts = time.mktime(date.timetuple())
    return ts


def _get_field_type(field):
    ftype = field.Type.replace('System.', '').lower()
    if 'int' in ftype:
        return 'int'
    elif 'datetime' == ftype:
        return 'datetime'
    elif 'string' == ftype:
        return 'str'
    else:
        return 'float'


def cleanup_na(val):
    if val == 'NA' or val is None or val == 'None' or val == 'null' or val == 'NULL' or \
            (val == float("Inf") or val == float("-Inf")):
        return 0
    return val


def preprocess_training_data(data, fields, deconstruct_ts=True):
    if fields is None:
        return data
    if isinstance(data, dict):
        for k in data:
            data[k] = preprocess_training_data(data[k], fields)
        return data
    cols_to_drop = set()
    for column in data:
        data[column] = data[column].apply(cleanup_na)
    for i, row in data.iterrows():
        for f in fields:
            if isinstance(f, dict):
                fname = f['name']
                ftype = f['type']
                encoding = f['encoding'] if 'encoding' in f else None
                is_key = f['is_key'] if 'is_key' in f else None
            else:
                continue
            if fname not in data:
                continue

            # We must convert timestamps to ints
            if ftype == 'datetime' and deconstruct_ts:
                val = row[fname]
                #
                if fname + '_year' not in data:
                    data[fname + '_year'] = 0
                data[fname + '_year'][i] = val.year
                #
                if fname + '_month' not in data:
                    data[fname + '_month'] = 0
                data[fname + '_month'][i] = val.month
                #
                if fname + '_day' not in data:
                    data[fname + '_day'] = 0
                data[fname + '_day'][i] = val.day
                #
                if fname + '_hour' not in data:
                    data[fname + '_hour'] = 0
                data[fname + '_hour'][i] = val.hour
                #
                if fname + '_minute' not in data:
                    data[fname + '_minute'] = 0
                data[fname + '_minute'][i] = val.minute
                cols_to_drop.add(fname)
    data.drop(list(cols_to_drop), axis=1, inplace=True)
    return data
