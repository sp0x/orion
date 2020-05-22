def has_func(functions_list, name):
    return len(list(x for x in functions_list if x.type == 'funcdef' and x.name.value == name)) > 0


def has_class(node_list, name):
    return len(list(x for x in node_list if x.type == 'classdef' and x.name.value == name)) > 0


def get_class(node_list, name):
    for x in node_list:
        if x.type == 'classdef' and x.name.value == name:
            return x


def get_first_function_name(node_list):
    if isinstance(node_list, list):
        for x in [xx for xx in node_list if xx.type == 'funcdef']:
            return x.name.value
    else:
        if node_list.type=='funcdef':
            return node_list.name.value
        else:
            return get_first_function_name(node_list.children)


def get_func(node_list, name):
    for x in node_list:
        if x.type == 'funcdef' and x.name.value == name:
            return x
