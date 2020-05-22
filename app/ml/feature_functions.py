import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta


def sum(elements, key):
    sum = 0
    for e in elements:
        val = e[key]
        if val is None:
            val = 0
        sum += val
    return sum


def std(elements, key):
    array_raw = [e[key] if e[key] is not None else 0 for e in elements]
    arr = np.array(array_raw)
    return arr.std()


def max(elements, key):
    array_raw = [e[key] if e[key] is not None else 0 for e in elements]
    arr = np.array(array_raw)
    max = arr.max()
    return max


def min(elements, key):
    array_raw = [e[key] if e[key] is not None else 0 for e in elements]
    arr = np.array(array_raw)
    return arr.min()


def skew(elements, key):
    from scipy.stats import skew
    array_raw = [e[key] if e[key] is not None else 0 for e in elements]
    arr = np.array(array_raw)
    return skew(arr)


def mean(elements, key):
    array_raw = [e[key] if e[key] is not None else 0 for e in elements]
    arr = np.array(array_raw)
    return arr.mean()


def mode(elements, key):
    from scipy.stats import mode
    array_raw = [e[key] if e[key] is not None else 0 for e in elements]
    arr = np.array(array_raw)
    return mode(arr)


def count(elements, key):
    elements_ = [e[key] if e[key] is not None else 0 for e in elements]
    return len(elements_) if len(elements) > 0 else 0


def day(elements, key):
    elements = [e[key] for e in elements if e[key] is not None]
    output = [x.day if isinstance(x, datetime) else 0 for x in elements] if len(elements) > 0 else 0
    return output


def year(elements, key):
    elements = [e[key] for e in elements if e[key] is not None]
    return [x.year if isinstance(x, datetime) else 0 for x in elements] if len(elements) > 0 else 0


def month(elements, key):
    elements = [e[key] for e in elements if e[key] is not None]
    return [x.month if isinstance(x, datetime) else 0 for x in elements] if len(elements) > 0 else 0


def weekday(elements, key):
    elements = [e[key] for e in elements if e[key] is not None]
    return [x.weekday() if isinstance(x, datetime) else 0 for x in elements] if len(elements) > 0 else 0


def hour(elements, key):
    elements = [e[key] for e in elements if e[key] is not None]
    return [x.hour for x in elements] if len(elements) > 0 else 0


def second(elements, key):
    elements = [e[key] for e in elements if e[key] is not None]
    return [x.second for x in elements] if len(elements) > 0 else 0


def field(elements, key):
    elements = [e[key] for e in elements if e[key] is not None]
    return [x for x in elements] if len(elements) > 0 else 0


def log(elements, key):
    array_raw = [e[key] if e[key] is not None else 0 for e in elements]
    arr = np.array(array_raw)
    ret = list(np.log(arr))
    ret = [0 if (x == float("Inf") or x == float("-Inf") or np.isinf(x)) else x for x in ret]
    return ret


def isnot(elements, key):
    elements = [e[key] for e in elements if e[key] is not None]
    return [not x for x in elements] if len(elements) > 0 else 0


def binary(left, right, key, elements):
    left_value = direct(left['name'], elements, left['key'], left['index'])
    if key == '=':
        return left_value == right
    elif key == '>':
        return left_value > right
    elif key == '<':
        return left_value < right
    elif key in ['not', '!=']:
        return left_value != right
    pass


def predicate(feature, elements):
    type = feature['name']
    base_feature = feature['key']
    # (base_feature['name'], elements, base_feature['key'], base_feature['index'])
    value = resolve(base_feature, elements)
    if type == 'in':
        return value in feature['in_values']
    elif type == 'num_unique':
        if not isinstance(value, list):
            value = [value]
        return len(set(value))
    else:
        print("Predicate feature not handled: ")
        print(feature)


def recursive(feature, elements):
    args = [feature['key']]
    argvals = []
    for arg in args:
        arg_val = resolve(arg, elements)
        argvals.append({'k': arg_val})
    module = sys.modules[__name__]
    op = feature['name'] if feature['name'] is not None else 'field'
    method = getattr(module, op) if hasattr(module, op) else None
    if method is None:
        raise NotImplementedError("Feature not implemented: " + str(op))
    feature_val = method(argvals, 'k')
    if isinstance(feature_val, (np.ndarray)):
        feature_val = list(feature_val)
    if isinstance(feature_val, list):
        feature_val = feature_val[0]
    return feature_val


def direct(op, elements, key, index=None):
    module = sys.modules[__name__]
    op = op if op is not None else 'field'
    op = op.lower()
    method = getattr(module, op) if hasattr(module, op) else None
    if index is not None:
        if index == 'first':
            elements = elements[:1]
        elif index == 'last':
            elements = elements[-1:]
    if method is None:
        raise NotImplementedError("Feature not implemented: " + str(op))
    assert method is not None
    if not isinstance(key, str):
        sub_elements = resolve(key, elements)
        if not isinstance(sub_elements, list):
            sub_elements = [sub_elements]
        tmp_elements = []
        for elem in sub_elements:
            tmp_elements.append({
                'k': elem
            })
        feature_val = method(tmp_elements, 'k')
    else:
        feature_val = method(elements, key)

    if index is not None:
        return feature_val[0]
    return feature_val


def first(param):
    return param[0] if isinstance(param, list) else param


def unique(param):
    if not isinstance(param, list):
        param = [param]
    return set(param)


def get_index_suffix(index):
    elem_suffix = ''
    if index is not None:
        if index == 'first':
            elem_suffix = '[:1]'
        elif index == 'last':
            elem_suffix = '[-1:]'
    return elem_suffix


def direct_template(op, key, index=None):
    """
    :param op:
    :param key:
    :param index:
    :return:
    """
    module = sys.modules[__name__]
    op = op if op is not None else 'field'
    op = op.lower()
    method = getattr(module, op) if hasattr(module, op) else None
    if method is None:
        raise Exception('No method specified or method could not be found! Op: ' + str(op if op is not None else ''))
    method = method.__name__
    elem_suffix = get_index_suffix(index)
    output = "{0}".format(method)
    if isinstance(key, dict):
        sub_f = gen_feature_expr(key)
        args = "({0}, '{1}')".format("elements" + elem_suffix, sub_f)
    else:
        args = "({0}, '{1}')".format("elements" + elem_suffix, key)
    output += args
    # if op == 'field':
    #    output = 'first(' + output + ')'
    return output

def custom_template(ftr):
    return ftr

#
def recursive_template(feature):
    base_ftr = feature['key']
    if is_field_fetch(base_ftr):
        ftr_key = base_ftr['key']
        if 'index' in base_ftr and base_ftr['index'] is not None:
            ftr_key += get_index_suffix(feature['index'])
        base_template = ftr_key
        base_template = '{0}'.format(base_template.strip('\'\"'))
        feature['key'] = base_template
        feature['type'] = 'direct'
        output = gen_feature_expr(feature)
    else:
        base_template = gen_feature_expr(base_ftr)
        base_template = "{0}".format(base_template)
        output = "[{'k': x} for i, x in enumerate(" + base_template + ")]"
        ftr_key = feature['name']
        module = sys.modules[__name__]
        op = ftr_key if ftr_key is not None else 'field'
        method = getattr(module, op) if hasattr(module, op) else None
        if method is None:
            raise Exception('No method specified!')
        method = method.__name__
        output = "ff.{0}({1}, 'k')".format(method, output)
    return output


def binary_template(left, right, key):
    left_template = direct_template(left['name'], left['key'], index=left['index'])
    if key == '=':
        op = '=='
    elif key == '>':
        op = '>'
    elif key == '<':
        op = '<'
    elif key in ['not', '!=']:
        op = '!='
    else:
        raise NotImplementedError()
    template = '{0} {1} {2}'.format(left_template, op, right)
    return template


def predicate_template(feature):
    type = feature['name']
    base_feature = feature['key']
    # (base_feature['name'], elements, base_feature['key'], base_feature['index'])
    value = gen_feature_expr(base_feature)
    template = ''
    if type == 'in':
        template = '{0} in {1}'.format(value, feature['in_values'])
    elif type == 'num_unique':
        template = 'len(unique({0}))'.format(value)
    else:
        raise Exception("Predicate feature not handled: " + str(feature))
    return template


def resolve(feature, elements):
    if feature['type'] == 'direct' and isinstance(feature['key'], dict):
        feature['type'] = 'recursive'

    if feature['type'] == 'direct':
        feature_val = direct(feature['name'], elements, feature['key'],
                             index=(feature['index'] if 'index' in feature else None))
    elif feature['type'] == 'binary':
        feature_val = binary(feature['left'], feature['right'], feature['op'], elements)
    elif feature['type'] == 'predicate':
        feature_val = predicate(feature, elements)
    elif feature['type'] == 'recursive':
        feature_val = recursive(feature, elements)
    else:
        raise NotImplementedError("Feature type could not be resolved")
    return feature_val


def is_field_fetch(ftr):
    return ftr['name'] is None and isinstance(ftr['key'], str)


def gen_feature_expr(feature):
    """
    Generates a feature expression from a feature dict.
    :param feature: {type: 'direct|recursive|binary|predicate'
    :return:
    """
    ftr_type = feature['type']
    func_name = feature['name'] if 'name' in feature else None
    if ftr_type == 'direct' and isinstance(feature['key'], dict):
        ftr_type = 'recursive'
    if ftr_type == 'direct':
        ftr_template = direct_template(func_name, feature['key'],
                                       index=(feature['index'] if 'index' in feature else None))
    elif ftr_type == 'binary':
        ftr_template = binary_template(feature['left'], feature['right'], feature['op'])
    elif ftr_type == 'predicate':
        ftr_template = predicate_template(feature)
    elif ftr_type == 'recursive':
        ftr_template = recursive_template(feature)
    elif ftr_type == 'custom':
        ftr_template = custom_template(feature)
    else:
        raise NotImplementedError("Feature type `" + ftr_type + "` could not be resolved")
    return ftr_template
