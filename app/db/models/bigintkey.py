import peewee as pw


class BigIntPrimaryKey(pw.PrimaryKeyField):
    field_type = 'BIGAUTO'
