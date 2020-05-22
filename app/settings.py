import os
from pymongo import MongoClient
import psycopg2
import boto3

AUTOML_BACKEND = os.environ.get('AUTOML_BACKEND', 'tpot')
default_scoring = {
    'regression': 'mean_squared_error',
    'classification': 'accuracy'
}


def get(name, default=None):
    """Gets a setting"""
    val = os.environ.get(name, default=default)
    return val


def get_log_file(filename):
    base = "/experiments/log"
    if not os.path.exists(base):
        os.makedirs(base)
    full_file = os.path.join(base, filename + '.log')
    return full_file


def get_db_settings():
    """Gets the settings for mongodb"""
    username = get('mongo_user')
    mpass = get('mongo_pass')
    port = get('mongo_port')
    if port is None:
        port = "27017"
    hostname = get('mongo_host')
    db = get('mongo_db')
    if db is None:
        db = "netvoid"
    auth_part = username if username is not None and mpass is not None else None
    prefix = "mongodb://"
    url = prefix + ((auth_part + "@") if auth_part is not None else "")
    url += hostname + ":" + port
    url += "/" + db
    if mpass is not None and username is not None:
        url += "?authSource=admin"
    return {
        'password': mpass,
        'url': url
    }


def get_s3_client():
    """

    :return:  BaseClient
    """
    import boto3
    return boto3.client('s3')


def get_s3():
    import boto3
    return boto3.resource('s3')


def get_db():
    db_name = get('mongo_db')
    mongo_url = get_db_url()
    client = MongoClient(mongo_url)
    return client[db_name]


def get_redis():
    import redis
    redis_host = get('REDIS_HOST')
    if redis_host is None or len(redis_host) == 0:
        redis_host = "localhost"
    redis_port = get('REDIS_PORT')
    if redis_port is None or len(redis_port) == 0:
        redis_port = "6379"
    redis_db = get('REDIS_DB')
    if redis_db is None or len(redis_db) == 0:
        redis_db = "0"
    r = redis.StrictRedis(host=redis_host, port=int(redis_port), db=int(redis_db), charset="utf-8",
                          decode_responses=True)
    return r


def get_postgre():
    con = psycopg2.connect(
        host=get('POSTGRES_HOST'),
        port=get('POSTGRES_PORT'),
        database=get('POSTGRES_DB'),
        user='postgres',
        password=get('POSTGRES_PASSWORD'))
    return con.cursor()


def empty(x):
    return x is None or len(x) == 0


def get_db_url():
    """Gets the settings for mongodb"""
    username = get('mongo_user')
    mpass = get('mongo_pass')
    port = get('mongo_port')
    if port is None or len(port) == 0:
        port = "27017"
    hostname = get('mongo_host')
    db = get('mongo_db')
    if db is None:
        db = "netvoid"
    auth_part = "{0}:{1}".format(username, mpass) if not empty(username) and not empty(mpass) else None
    prefix = "mongodb://"
    url = prefix + ((auth_part + "@") if auth_part is not None else "")
    url += hostname + ":" + port
    url += "/" + db
    if not empty(mpass) and not empty(username):
        url += "?authSource=admin"
    return url
