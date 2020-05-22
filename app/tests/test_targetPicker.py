from unittest import TestCase
import mongomock
from datetime import datetime, timedelta

from ml.targets import TargetPicker
from tests.test_helpers import timeify


class TestTargetPicker(TestCase):
    def setUp(self):
        from tests.test_helpers import fill_db
        db = mongomock.MongoClient().db
        key = 'collectionIntelflows'
        self.collection = fill_db(db, key, 'intelflows.cities.csv', lambda x: timeify(x, 'timestamp'), cnt=100)
        self.targets = [{
            'column': 'pm10',
            'scoring': 'auto', #default is auto
            'task_type': 'auto', #default is auto
            'constraints': [
                {
                    'type': 'time',
                    'key': 'timestamp',
                    'after': {
                        'hours': 1
                    }
                },
                {'key': 'city_id'},
                {'key': 'device_ID'},
            ]
        }]

    def test_get_target_query(self):
        targets_picker = TargetPicker(self.targets, self.collection)
        t = datetime.now()
        rec = {
            'city_id': 1,
            'device_ID': 33,
            'pm10': 1,
            'timestamp': t
        }
        query = targets_picker.get_target_query("pm10", rec)
        self.assertTrue(query['query']['city_id'] == 1)
        self.assertTrue(query['query']['device_ID'] == 33)
        self.assertTrue(query['extras']['sorting'][0][0] == 'timestamp')
        tdiff = query['query']['timestamp']['$gte'] - t
        self.assertTrue(tdiff == timedelta(hours=1))

    def test_get_target_values(self):
        targets_picker = TargetPicker(self.targets, self.collection)
        first_item = self.collection.find_one()
        t_vals = targets_picker.get_target_values(first_item)
        self.assertTrue(t_vals['pm10'] == 14.0)
