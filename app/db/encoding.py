import numpy as np
import pandas as pd
from db.models import FieldExtra


def bin_to_int(b):
    b = b[1:]
    val = int(b, 2)
    return val


def int_to_bin(i):
    binary = "1{0:08b}".format(i)
    return binary


def __decode_binary_array(array):
    s_array = [bin_to_int(x) for x in array]
    feature_val = np.average(s_array)
    return feature_val


class EncodingHelper:
    def __init__(self, data_flags=None):
        self.encoders = {
            'binaryintid': EncodingHelper.__decode_binary
        }
        self.decoders = {
            'binaryintid': EncodingHelper.__dec_bin_field
        }
        self.val_decoders = {
            'binaryintid': EncodingHelper.__dec_bin_field_val
        }
        self.val_encoders = {
            'binaryintid': EncodingHelper.__enc_bin_field_val
        }
        self.data_flags = data_flags

    def has(self, field_name):
        if self.data_flags is None:
            return False
        return field_name in [x['name'] for x in self.data_flags['fields']]

    def get_field(self, name):
        for x in self.data_flags['fields']:
            if x['name'] == name:
                return x
        return None

    def get_decodable_fields(self, data_flags, columns):
        output = {}
        for field in data_flags['fields']:
            fld_name = field['name']
            if 'encoding' not in field or fld_name not in columns:
                continue
            field_encoding = field['encoding']
            if field_encoding is None or len(field_encoding) == 0 or field_encoding not in self.val_decoders:
                continue
            decoder = self.val_decoders[field_encoding]
            output[fld_name] = decoder
        return output

    @staticmethod
    def __decode_binary(df, field):
        def f(row, index):
            return str(row[field])[index + 1]

        for i in range(0, 8):
            df["{0}{1}".format(field, i)] = df.apply((lambda x: f(x, i)), axis=1)
        del df[field]  # Delete the encoded value
        return df

    @staticmethod
    def __dec_bin_field(df, field):
        if isinstance(df, pd.DataFrame):
            def f(row):
                return bin_to_int(str(row[field]))

            df[field] = df.apply((lambda x: f(x)), axis=1)
            return df
        else:
            # must be ndarray
            def f(x):
                return bin_to_int(str(x))

            f = np.vectorize(f, otypes=[np.float])
            output = f(df)
            return output

    @staticmethod
    def __dec_bin_field_val(value):
        return bin_to_int(str(value))

    @staticmethod
    def __enc_bin_field_val(value, field):
        # TODO: Make this a transaction..
        encoded_count = FieldExtra.select().where(FieldExtra.FieldId == field['id']).count()
        key = int_to_bin(encoded_count)
        existing_field = FieldExtra.select(FieldExtra).where(
            (FieldExtra.FieldId == field['id']) & (FieldExtra.Key == key)).first()
        if existing_field is not None:
            return existing_field.Key
        encoded = FieldExtra()
        encoded.FieldId = field['id']
        encoded.Value = value
        encoded.Key = key
        encoded.Type = 0
        encoded.save()
        return key

    def decode_raw(self, df, data_flags):
        fields = data_flags['fields']
        sz = len(df)

        field_decs = self.get_decodable_fields(data_flags, list(df.columns))
        for index, row in df.iterrows():
            for dec_fld in field_decs:
                dec = field_decs[dec_fld]
                raw_val = row[dec_fld]
                dec_val = dec(raw_val)
                row[dec_fld] = dec_val
            df[index] = row
        return df

    def decode_row(self, row, fields):
        for field in fields:
            fld_name = field['name']
            if fld_name not in row:
                continue
            if 'encoding' not in field:
                continue
            field_encoding = field['encoding']
            if field_encoding is None or len(field_encoding) == 0 or field_encoding not in self.val_decoders:
                continue
            value = row[fld_name]
            decoder = self.val_decoders[field_encoding]
            d_value = decoder(value)
            row[fld_name] = d_value
        return row

    def get_field(self, field_name):
        for f in self.data_flags['fields']:
            if f['name'] == field_name:
                return f

    def decode_plaintext(self, value, field):
        if isinstance(field, str):
            field = self.get_field(field)
        fld_name = field['name']
        field_id = field['id']
        if 'encoding' not in field:
            return value
        field_encoding = field['encoding']
        if field_encoding is None or len(field_encoding) == 0 or field_encoding not in self.val_encoders:
            return value
        if field_encoding is None or len(field_encoding) == 0 or field_encoding not in self.val_decoders:
            return value
        key_encoder = self.val_encoders[field_encoding]
        encoded_key = key_encoder(value, field)
        field = FieldExtra.select(FieldExtra).where(
            (FieldExtra.FieldId == field_id) & ( FieldExtra.Key == encoded_key)).first()
        return field.Value

    def decode_raw_col(self, df, field):
        field_data = self.get_field(field)
        if 'encoding' not in field_data:
            return df
        encoding = field_data['encoding']
        if encoding is None or len(encoding) == 0:
            return df
        if encoding not in self.decoders:
            raise KeyError("Encoding `{0}` not supported.".format(encoding))
        else:
            decoder = self.decoders[encoding]
            df = decoder(df, field)
        return df

    def encode_frame_field(self, df, fld):
        """

        :param df:
        :param fld:
        :return:
        """
        field_name = fld['name']
        field_encoding = fld['encoding']
        if field_encoding not in self.encoders:
            raise KeyError("Encoding `{0}` not supported.".format(field_encoding))
        else:
            encoder = self.encoders[field_encoding]
            df = encoder(df, field_name)
        return df

    def fill_encoded(self, df: pd.DataFrame, fld):
        field_name = fld['name']
        field_encoding = fld['encoding']
        field_id = fld['id']
        for row in df.itertuples():
            plaintext = getattr(row, field_name)
            field = FieldExtra.select(FieldExtra).where(
                (FieldExtra.FieldId == field_id) & (FieldExtra.Value == plaintext)).first()
            if field is not None:
                df.set_value(row.Index, field_name, field.Key)
            else:
                encoder = self.val_encoders[field_encoding]
                df = encoder(plaintext, fld)
        return df
