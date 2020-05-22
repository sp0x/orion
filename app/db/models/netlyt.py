import peewee as pw
from settings import get
from playhouse.postgres_ext import PostgresqlExtDatabase, JSONField

db = PostgresqlExtDatabase(
    host=get('POSTGRES_HOST'),
    port=get('POSTGRES_PORT'),
    database=get('POSTGRES_DB'),
    user='postgres',
    password=get('POSTGRES_PASSWORD')
)


class NetlytModel(pw.Model):
    class Meta:
        database = db
