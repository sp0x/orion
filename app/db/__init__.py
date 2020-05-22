import settings
from .data_handlers import *




def fetch_record(query, collection):
    db = settings.get_db()
    col = db[collection]
    doc = col.find_one(query)
    return doc


def get_full_data(input: dict, integration_details: dict):
    from ml import DataException
    is_partial = False
    fields_ = [x['name'] for x in integration_details['schema']]
    for f in input:
        if f not in fields_:
            is_partial = True
            break
    if is_partial:
        if 'id' in integration_details and integration_details['id'] is not None and len(integration_details['id']) > 0:
            if integration_details['id'] not in input:
                raise DataException("Can't use partial input when input doesn't contain the model's"
                                    " `{0}` key field.".format(integration_details['id']))
            id_val = input[integration_details['id']]
            collection = integration_details['collection']
            query = {integration_details['id']: id_val}
            input = fetch_record(query, collection)
        else:
            raise DataException("Can't use partial input when model doesn't have an Id field.")
    return input


def get_model_build(build_id):
    from settings import get_redis
    redis = get_redis()
    model_hash = redis.hgetall("builds:published:{0}".format(build_id))
    return model_hash
