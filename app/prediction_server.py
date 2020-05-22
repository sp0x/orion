from flask import Flask, request, abort, jsonify, Response
from flask_cors import CORS
import os
import threading
import settings
import logging
from pandas.io.json import json_normalize
from datetime import datetime, timedelta

from db import get_full_data
from ml import DataException, MlScript, name_from_model
from ml.prediction import PredictionHelper
from db.models import db

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def authorize_request(req, build_id, body):
    from db import get_model_build
    secret = req.headers.get('secret', default=None)
    api_key = request.headers.get('key', default=None)
    bad_output = False, None, None
    if secret is None or api_key is None:
        return bad_output
    model_build = get_model_build(build_id)
    build_key = str(model_build.get('key', None))
    build_secret = str(model_build.get('sec', None))
    if model_build is None:
        return bad_output
    authorized = build_key == api_key and build_secret == secret
    # Todo: improve this with HMAC and request validation, like it is in Netlyt.
    if not authorized:
        return bad_output
    else:
        model_id = int(model_build['model_id'])
        return True, {
            'details': {
                'build_id': build_id,
                'model_id': model_id,
                'user': model_build['user']
            },
        }


def get_model_target_cols(model):
    from db.models import get_model_target_fields
    target_fields = get_model_target_fields(model)
    cols = [x.Name for x in target_fields]
    return cols


def create_prediction_targets_definition(model, integration):
    columns = get_model_target_cols(model)
    fbuilder_config = {
        'columns': columns,
        'constraints': []
    }
    return fbuilder_config


def throttle(model_details):
    from input import Throttler
    throttler = Throttler(model_details)
    return throttler.throttle()


def log_performance(model_details, ip):
    #    mp = ModelPerformance.get(ModelPerformance.ModelId == m_id)
    from input import Throttler
    throttler = Throttler(model_details)
    throttler.save_model_performance(ip)



class PredictionServer(threading.Thread):

    def __init__(self):
        super(PredictionServer, self).__init__()

        self.flask = FlaskAppWrapper('prediction')
        self.__register_endpoints([
            ('/<build_id>', 'predict', self.predict, 'POST'),
            ('/status', 'status', self.status, 'GET'),
            ('/ml_status', 'ml_status', self.ml_status, 'GET'),
        ])
        self.port = 80
        PORT = settings.get("PORT", default=None)
        if not PORT:
            self.port = 80
        else:
            self.port = PORT
        HOST = os.environ.get("HOST", default=None)
        if not HOST:
            self.host = '0.0.0.0'
        else:
            self.host = HOST

    def __register_endpoints(self, endpoints):
        for endpoint in endpoints:
            self.flask.add_endpoint(endpoint=endpoint[0], endpoint_name=endpoint[1], handler=endpoint[2],
                                    method=endpoint[3])
        return self

    def status(self):
        response = Response(status=200, headers={})
        return response

    def ml_status(self):
        response = Response(status=200, headers={})
        return response

    def predict(self, build_id):
        from ml import load_ml_script
        from ml.classifiers import Experiment
        import ml.cleanup
        import ml.preprocessing
        from input import format_prediction_input

        logging.info(str(build_id))
        response = Response(status=200, headers={})
        if request.data:
            # try:
            data = request.get_json(silent=False)
            auth_result = authorize_request(request, build_id, data)
            logging.info(str(auth_result))
            authorized, request_details = auth_result[0], auth_result[1]
            if not authorized:
                abort(401, "Unauthorized.")
            build_details = request_details['details']
            if not throttle(build_details):
                abort(429, "You have exceeded your monthly request quota please contact sales@netlyt.io")
            script = load_ml_script(build_details['user'], name_from_model(build_details['model_id']))
            threading._start_new_thread(log_performance, (build_details, request.remote_addr))
            data_rows = []
            if 'data' in data:
                for row in data['data']:
                    formatted = format_prediction_input(input=row, script=script)
                    if formatted is not None:
                        data_rows.append(formatted)
            else:
                formatted = format_prediction_input(input=data, script=script)
                if formatted is not None:
                    data_rows.append(formatted)
            if len(data_rows) == 0:
                abort(404, 'No data for predictions could be found.')
                return
            df = json_normalize(data_rows)
            df = ml.cleanup.cleanup_key_fields(df, script)
            # Todo: find a way to not load the model on every request..
            pred_helper = PredictionHelper.from_script(script)
            df = pred_helper.encode_df(df)
            df = ml.preprocessing.preprocess_training_data(df, script.input_flags()['fields'], deconstruct_ts=False)
            try:
                result = pred_helper.predict_raw(df, len(data_rows))
                return jsonify({
                    'result': result
                })
            except DataException as e:
                response = jsonify({'success': False, 'error': str(e)})
                response.status_code = 500
        else:
            abort(500, "Badly formatted data.")
        return response

    def run(self):
        log.info('Prediction server starting on {0}:{1}'.format(self.host, self.port))
        self.flask.run(host=self.host, port=self.port)


class EndpointAction(object):
    def __init__(self, action):
        self.action = action

    def __call__(self, *args, **kwargs):
        response = self.action(*args, **kwargs)
        return response


class FlaskAppWrapper(object):
    app = None

    def __init__(self, name):
        self.app = Flask(name)
        CORS(self.app, supports_credentials=True)

    def run(self, host, port):
        self.app.run(host=host, port=port)

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, method='GET'):
        self.app.add_url_rule(endpoint, endpoint_name, view_func=EndpointAction(handler), methods=[method, ])
