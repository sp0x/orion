from input.throttle import Throttler
from datetime import datetime
from ml import MlScript

def get_field_flags(field, script: MlScript):
    flags = script.input_flags()
    fields = flags['fields']
    for f in fields:
        if f['name'] == field:
            return f


def format_prediction_input(script: MlScript, input: dict) -> object:
    """
    Formats a dict so all it's values are numeric
    :param script: dict A script that describes the prediction task
    :param input: dict
    :return:
    """
    from db import get_full_data
    flags = script.input_flags()
    tscol = flags['collection']['timestamp']
    id_col = flags['collection']['index']
    input = get_full_data(input, {
        'schema': flags['fields'],
        'id': id_col,
        'collection': flags['collection']['key']
    })
    if tscol is not None and isinstance(input[tscol], int):
        timestamp_ = input[tscol]
        f = 100
        if len(str(timestamp_)) == len("1525103750000"):
            f = 1000
        if len(str(timestamp_)) == len("15275123770000"):
            f = 10000
        input[tscol] = datetime.fromtimestamp(timestamp_ / f)
    # Check for badly formatted input, $date timestamps for example.
    for col in input:
        value = input[col]
        if isinstance(value, dict):
            column_info = get_field_flags(col, script)
            if column_info is not None and column_info['type'] == 'datetime':
                ts_val = value['$date'] / 1000
                input[col] = datetime.fromtimestamp(ts_val)
    return input
