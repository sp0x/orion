import os
from datetime import timedelta, datetime
import pickle
import glob
import json
import numpy as np
import scipy as sp
import math
import pandas as pd
import os
import re
import zipfile

re_phone = re.compile("^\+?(([\(\) \\\/\d]{1,10})(-| )?){2,5}$")


def load_module(name, location):
    """
    Loads a ml script from an absolute path.
    :param name:
    :param location:
    :param args:
    :param kwargs:
    :return:
    """
    import importlib.util
    import importlib
    import ntpath
    if name is None:
        name = ntpath.basename(location)
    if '' not in importlib.machinery.SOURCE_SUFFIXES:
        importlib.machinery.SOURCE_SUFFIXES.append('')
    spec = importlib.util.spec_from_file_location(name, location)
    script_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_mod)
    return script_mod

def file_exists_in_dir(FOLDER_PATH='../', FILE_NAME=__file__):
    return os.path.isdir(FOLDER_PATH) \
           and os.path.isfile(os.path.join(FOLDER_PATH, FILE_NAME))


def file_exists(file_path):
    return os.path.isfile(file_path)


def is_phone_number(txt):
    return re_phone.match(txt)

def zipdir_abs(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def zipdir(path, ziph):
    # ziph is zipfile handle
    lenDirPath = len(path)
    for root, _, files in os.walk(path):
        for file in files:
            filePath = os.path.join(root, file)
            ziph.write(filePath, filePath[lenDirPath:])


def mkdir_p(path):
    try:
        if os.path.isdir(path):
            return
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def update_lines(string, lambda_mod):
    """
    Modify lines in string by applying a lambda to them.
    :param string:
    :param lambda_mod:
    :return:
    """
    from io import StringIO
    stri = StringIO(string)
    output = StringIO()
    while True:
        nl = stri.readline()
        if nl != '':
            nl = nl.rstrip()
            nl = lambda_mod(nl)
            output.write(nl + "\n")
        else:
            break
    return output.getvalue()


def resolve_type(dtype, format_as=None):
    """

    :param dtype:
    :param format_as:
    :return:
    """
    if not isinstance(dtype, object) or format_as == str:
        return 'str'
    if isinstance(dtype, datetime) or np.issubdtype(dtype, np.datetime64) or format_as == datetime:
        return 'datetime'
    elif dtype is np.dtype('O'):
        return 'str'
    return str(dtype)


def todf(ndarray):
    df = pd.DataFrame(data=ndarray[1:, 1:],
                      index=ndarray[1:, 0],
                      columns=ndarray[0, 1:])
    return df


def regression_report(y_true, y_pred, estimator=""):
    """
        Helper function for error outputs.
        
        Parameters
        ----------
        y_true : numpy.1darray
            true labels for test data
        
        y_pred : numpy.1darray
            predicted labels for test data
        
        estimator : str
            name of estimator (for output purposes)
        
        Returns
        -------
        None
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    output = "%s mean absolute error:  %s\n" % (estimator, mean_absolute_error(y_true, y_pred))
    output = output + "%s mean squared error: %s" % (estimator, mean_squared_error(y_true, y_pred))
    return output


def abs_path(fl=None):
    curr_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    if not fl:
        return curr_dir
    abs_file_path = os.path.join(curr_dir, fl)
    return abs_file_path


def reduce_dim(x, d):
    pad_size = math.ceil(float(x.size) / d) * d - x.size
    x_padded = np.append(x, np.zeros(pad_size) * np.NaN)
    return sp.nanmean(x_padded.reshape(-1, d), axis=0)


def hasmethod(o, method):
    op = getattr(o, method, None)
    return callable(op)


flatten = lambda l: [i for sl in l for i in sl]


def par_path(fl=None):
    path = abs_path()
    path = os.path.abspath(os.path.join(path, os.pardir))
    return os.path.join(path, fl)


def proportion(data, item):
    return list(data).count(item) / float(len(data))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def parse_timespan(span):
    duration = span.split(":")
    hours = float(duration[0])
    mins = float(duration[1])
    seconds = float(duration[2])
    duration = timedelta(hours=hours, minutes=mins, seconds=seconds)
    return duration


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def latest_file(filepath):
    files = glob.glob(filepath + "*")
    last_file = max(files, key=os.path.getctime)
    return last_file


def load(path):
    out = None
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out


def loads(byts):
    out = pickle.loads(byts)
    return out


def load_json(path):
    out = None
    with open(path, 'r') as f:
        out = json.load(f)
    return out
