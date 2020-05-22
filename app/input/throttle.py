from datetime import datetime
from settings import get_redis


class Throttler:

    def __init__(self, build_details):
        self.build = build_details

    def save_model_performance(self, ip):
        redis = get_redis()
        day = datetime.today().weekday()
        day_of_month = datetime.now().day
        tsnow = datetime.now()
        key = "rates:current:{0}".format(self.build['user'])
        redis = get_redis()
        cr_usage = redis.hgetall(key)
        if cr_usage is None or len(cr_usage) == 0:
            cr_usage = {
                'weekly_usage': 0,
                'monthly_usage': 0,
                'daily_usage': 0
            }
        cr_usage['weekly_usage'] = int(cr_usage['weekly_usage']) + 1
        cr_usage['monthly_usage'] = int(cr_usage['monthly_usage']) + 1
        cr_usage['daily_usage'] = int(cr_usage['daily_usage']) + 1
        cr_usage['day'] = day
        cr_usage['day_of_month'] = day_of_month
        cr_usage['ts'] = tsnow
        cr_usage['ip'] = ip
        for el_key in cr_usage:
            value = cr_usage[el_key]
            redis.hset(key, el_key, value)

    def throttle(self):
        redis = get_redis()
        key = "rates:allowed:{0}".format(self.build['user'])
        user_rates_allowed = redis.hgetall(key)
        current_usage = self.current_rate()
        if int(current_usage['weekly_usage']) > int(user_rates_allowed['weekly_usage']):
            return False
        if int(current_usage['monthly_usage']) > int(user_rates_allowed['monthly_usage']):
            return False
        if int(current_usage['daily_usage']) > int(user_rates_allowed['daily_usage']):
            return False
        return True

    def current_rate(self):
        redis = get_redis()
        key = "rates:current:{0}".format(self.build['user'])
        output = redis.hgetall(key)
        if len(output) == 0:
            output = {
                'weekly_usage': 0,
                'monthly_usage': 0,
                'daily_usage': 0
            }
        return output
