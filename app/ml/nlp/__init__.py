# from langdetect import DetectorFactory
# from langdetect.lang_detect_exception import LangDetectException
# from langdetect.detector_factory import create_detector

# Testing..
from ml.nlp.detector_factory import DetectorFactory, create_detector
from ml.nlp.lang_detect_exception import LangDetectException
from ml.nlp.cnndetector import predict_batch, predict

DetectorFactory.seed = 0
_detector = create_detector()


def langdetect_detect(col_val):
    _detector.reset()
    if not isinstance(col_val, str):
        raise LangDetectException(col_val, 'Not a string')
    _detector.append(col_val)
    result = _detector.detect()
    return result


def langdetect_cnn(col_val):
    res = predict(col_val, True)
    return res[0]


def langdetect_cnn_batch(values, raw_text=True):
    res = predict_batch(values, raw_text)
    return res


def get_df_gen(df, columns):
    for index, row in df.iterrows():
        for column in columns:
            col_val = row[column]
            yield col_val


def analyze_column_language_batch(df, columns, detector_fn=langdetect_cnn_batch):
    """

    :param df:
    :param columns:
    :param detector_fn:
    :return:
    """
    lang_stats = {
        'by_cols': {}
    }
    if isinstance(columns, str):
        columns = [columns]
    val_gen = get_df_gen(df, columns)
    predictions = detector_fn(val_gen)
    index = 0
    col_index = 0
    for p in predictions:
        predicted_language = p[0]
        row = df.iloc[index]
        # column_value = row[col_index]
        column = columns[col_index]

        df.loc[index, column + ".lng"] = predicted_language
        if predicted_language not in lang_stats:
            lang_stats[predicted_language] = 0
        if column not in lang_stats['by_cols']:
            lang_stats['by_cols'][column] = {}
        if predicted_language not in lang_stats:
            lang_stats[column][predicted_language] = 0
        if predicted_language not in lang_stats['by_cols'][column]:
            lang_stats['by_cols'][column][predicted_language] = 0
        lang_stats[predicted_language] += 1
        lang_stats['by_cols'][column][predicted_language] += 1
        # Switch to the next row or col
        col_index += 1
        if col_index >= len(columns):
            col_index = 0
            index += 1
    return lang_stats


def analyze_column_language(df, columns, detector_fn=langdetect_detect):
    """
    Goes over each column and row and tries to figure out it's language
    :param detector_fn:
    :param df: Dataframe to analyze
    :param columns: The columns to detect the language for
    :return: A dict of { language: occurances }
    """
    lang_stats = {
        'by_cols': {}
    }
    if isinstance(columns, str):
        columns = [columns]
    for index, row in df.iterrows():
        for column in columns:
            col_val = row[column]
            try:
                lang = detector_fn(col_val)
            except LangDetectException as e:
                lang = 'INV'
            df.loc[index, column + ".lng"] = lang
            if lang not in lang_stats:
                lang_stats[lang] = 0
            if column not in lang_stats['by_cols']:
                lang_stats['by_cols'][column] = {}
            if lang not in lang_stats:
                lang_stats[column][lang] = 0
            if lang not in lang_stats['by_cols'][column]:
                lang_stats['by_cols'][column][lang] = 0
            lang_stats[lang] += 1
            lang_stats['by_cols'][column][lang] += 1
    return lang_stats
