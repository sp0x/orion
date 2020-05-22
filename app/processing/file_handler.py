import pandas as pd
import io
import boto3
import numpy as np
from processing.frames import DataframeParser, frame_src_type
from stat_utils import freq, hist
import settings


# Example usage
# pd_read_csv_s3("s3://my_bucket/my_folder/file.csv", skiprows=2)
def pd_read_csv_s3(path, *args, **kwargs):
    path = path.replace("s3://", "")
    bucket, key = path.split('/', 1)
    s3_client = settings.get_s3_client()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    if key.endswith('.csv'):
        kwargs.update({'nrows': 10000})
        return pd.read_csv(io.BytesIO(obj['Body'].read()), *args, **kwargs)
    elif key.endswith('.json'):
        return pd.read_json(io.BytesIO(obj['Body'].read()), *args, **kwargs)


@frame_src_type('collection')
class CollectionFrameParser(DataframeParser):
    def __init__(self, collection_name):
        from db.data_handlers import MongoBufferedReader
        super().__init__()
        mongo_reader = MongoBufferedReader()
        df = mongo_reader.read_unlimited_as_pandas_frame(collection_name)
        if '_id' in df:
            df = df.drop(['_id'], axis=1)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any')
        self.df = df


@frame_src_type('file')
class FileFrameParser(DataframeParser):
    def __init__(self, file_path):
        from db.data_handlers import MongoBufferedReader
        super().__init__()
        df = pd.read_csv(file_path)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any')
        self.df = df


@frame_src_type('s3')
class S3FrameParser(DataframeParser):
    def __init__(self, fl):
        super().__init__()
        df = pd_read_csv_s3(fl)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any')
        self.df = df
    #
    #
    # def get_summary(self):
    #     scheme = {k:str(v) for k,v in dict(self.df.dtypes).items()}
    #     summary = {'scheme':scheme}
    #     desc = {}
    #     for col in self.df:
    #         if scheme[col] == 'bool' or scheme[col] == 'object':
    #             desc[col] = freq(self.df, col)
    #         elif 'datetime' in scheme[col]:
    #             desc[col] ={'type':'range', 'values':{'min':str(self.df[col].min()), 'max':str(self.df[col].max())}}
    #         else:
    #             desc[col] = hist(self.df, col,'fd')
    #     summary['col_desc'] = desc
    #     return summary
