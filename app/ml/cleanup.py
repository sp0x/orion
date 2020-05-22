from ml import MlScript


def clear_data_for_features(df):
    if '_id' in df:
        df.drop('_id', axis=1, inplace=True)
    return df


def cleanup_premade_features(df):
    if '_id' in df:
        df.drop('_id', axis=1, inplace=True)
    return df


def clean_data_for_prediction(df):
    if '_id' in df:
        df.drop('_id', axis=1, inplace=True)
    return df


def drop_targets(targets, df):
    from ml.feature_extraction import get_target_column_name
    for t in [get_target_column_name(x) for x in targets]:
        if t in df:
            df.drop(t, axis=1, inplace=True)
    for t in [x for x in targets]:
        if t in df:
            df.drop(t, axis=1, inplace=True)
    return df


def clean_data_for_training(df_or_dictdf, fields=None):
    if isinstance(df_or_dictdf, dict):
        for k in df_or_dictdf:
            df_or_dictdf[k] = clean_data_for_training(df_or_dictdf[k], fields)
    else:
        df_or_dictdf = df_or_dictdf.dropna(axis=0, how='any')
        if fields is not None:
            for f in fields:
                if f['is_key']:
                    df_or_dictdf.drop(f['name'], axis=1, inplace=True)
    return df_or_dictdf


def cleanup_key_fields(df, script: MlScript):
    flags = script.input_flags()
    index = flags['collection']['index']
    if index is not None and len(index) > 0:
        df = df.drop(index, axis=1)
    return df
