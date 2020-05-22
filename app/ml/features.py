def get_custom_feature_name(ftr):
    from code_generation import get_first_function_name
    content = ftr['content'] if 'content' in ftr else None
    if content is None:
        return None
    import parso
    structure = parso.parse(content)
    name = get_first_function_name(structure)
    return name


def get_feature_name(ftr):
    name = ftr['title'] if ('title' in ftr and ftr['title'] is not None and len(ftr['title']) > 0) else \
        (get_custom_feature_name(ftr) if ftr['type'] == 'custom' else None)
    if name is None:
        raise Exception('All features  must have a name.')
    return name
