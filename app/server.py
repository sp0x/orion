from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import sys
import threading
from functools import partial
import zmq
import logging

from code_generation.project import Project
from constants import *
from ml import classifiers
from ml.classifiers import Experiment, build, build_scripted, train
from db.models import db
from utils import hasmethod
import traceback
import settings
from processing import get_frame_parser
import pydevd
from ml import ModelCache

# import ptvsd
# ptvsd.enable_attach(secret='dev', address=('0.0.0.0', 3000))

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_targets(msg):
    from ml.targets import parse_targets
    from ml.optimizers import get_ml_backend
    props = msg['params']
    targets = props['targets']
    collections = props['collections']
    target_types = parse_targets(targets, collections)
    return target_types


class Server(threading.Thread):
    def __init__(self):
        super(Server, self).__init__()
        self.context = zmq.Context()
        self.in_stream = self.context.socket(zmq.PULL)
        self.out_stream = self.context.socket(zmq.PUSH)
        self.event_stream = self.context.socket(zmq.PUSH)
        self.event_stream.bind("tcp://%s:%d" % (LISTEN_ADDR, EVENTS_PORT))
        self.in_stream.bind("tcp://%s:%d" % (LISTEN_ADDR, INPUT_PORT))
        self.out_stream.bind("tcp://%s:%d" % (LISTEN_ADDR, OUTPUT_PORT))
        self.is_running = True
        self.cond = threading.Lock()
        self.tpool = ThreadPoolExecutor(max_workers=10)
        self.ppool = multiprocessing.Pool(multiprocessing.cpu_count() or 4)
        cache_size = os.environ.get('MCACHE_SIZE', 5)
        self.cache = ModelCache(cache_size)

    def shutdown(self):
        with self.cond:
            self.is_running = False
            print("Waiting for all running experiments to finish executing")
            self.ppool.close()
            self.tpool.shutdown(wait=True)
            self.ppool.join()

    def reply(self, msg):
        try:
            self.out_stream.send_json(msg)
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def notify(result, params, server):
        try:
            server.event_stream.send_json(dict(result=result, params=params))
        except Exception as e:
            print(e)

    @staticmethod
    def on_features_generated(result, server, task):
        log.info("Features generated for model " + str(task.ModelId))
        with db:
            task.Status = 1
            task.save()
        Server.notify(result, server=server, params={
            'task_id': task.Id,
            'command': GEN_FEATURES,
            'model_id': task.ModelId
        })

    @staticmethod
    def on_training_complete(result, server, model_id, tasks):
        result_obj = result.result()
        log.info("Training complete generated model[{0}]: ".format(model_id))
        Server.notify(result_obj, server=server, params={
            'command': TRAIN,
            'model_id': model_id,
            'tasks': tasks
        })

    def make_prediction(self, msg):
        m_id = msg.get('model_id')
        p_type = msg.get('p_type', 'proba')
        data = msg.get('data')
        company = msg.get('company')
        model = self.cache.fetch(m_id, company)
        predictor = model['model']
        if p_type == 'proba' and hasmethod(predictor, 'predict_proba'):
            # res= model['model'].predict_proba(data)
            res = self.ppool.map(model['model'].predict_proba, data, 1000)
        elif p_type == 'log_proba' and hasmethod(predictor, 'predict_log_proba'):
            # res = model['model'].predict_log_proba(data)
            res = self.ppool.map(model['model'].predict_log_proba, data, 1000)
        else:
            # res = model['model'].predict(data)
            res = self.ppool.map(model['model'].predict, data, 1000)
        return {'results': {'value': res, 'model': model['type']}}

    def train(self, msg):
        #
        params = msg.get('params')
        tasks = params['tasks']

        model_id = params.get('model_id')
        log.info("Training with: " + str(params))
        if params['script'] is not None and params['script']['code'] is not None:
            exp = build_scripted(params)
        else:
            exp = build(params)

        callback = partial(Server.on_training_complete, server=self,
                           model_id=model_id,
                           tasks=tasks)
        future = self.tpool.submit(train, exp)
        future.add_done_callback(callback)
        return {'ids': exp.get_model_ids(), 'tids': exp.target_ids}

    def gen_features(self, msg):
        from db.data_handlers import MongoBufferedReader
        params = msg.get('params')
        model_id = params.get('model_id')
        from ml.feature_extraction import FeaturesBuilder
        log.info("Generating features: " + str(params))
        # we dont create tasks anymore..
        # with db:
        #     task = FeatureGenerationTasks.create(Status=0, ModelId=model_id)
        #     task.save()
        params['db'] = settings.get_db()
        fe = FeaturesBuilder(params, MongoBufferedReader)
        callback = partial(Server.on_features_generated, server=self, task=task)
        self.ppool.apply_async(fe.generate_features, callback=callback)
        return {'task_id': task.Id}

    @staticmethod
    def analyze_file(msg):
        """

        :param msg:
        :return:
        """
        props = msg["params"]
        src_type = props["src_type"]
        source = props['src']
        formatting = props['formatting']
        fp = get_frame_parser(src_type, source)
        resp = {}
        resp.update({'file_summary': fp.get_summary()})
        return resp


    def import_model(self, msg):
        from ml.storage import import_model
        result = import_model(msg['params'])
        return {
            'result': result
        }

    def create_script(self, msg):
        from code_generation import generate_script
        params = msg["params"]
        script = generate_script(params)
        return {
            'code': script
        }

    def generate_project(self, msg):
        """
        :param msg:
        :return: A url to a model file.
        """
        params = msg["msg"]
        project = Project(params['client'], params['name'])
        project.write_script(params['script']['code'])
        project.move_temp_assets()
        proj_zip = project.get_zip()
        return {
            'project': proj_zip
        }

    def run(self):
        log.info("Pull on tcp://%s:%d" % (LISTEN_ADDR, INPUT_PORT))
        log.info("Push on tcp://%s:%d" % (LISTEN_ADDR, OUTPUT_PORT))
        log.info("Events push on tcp://%s:%d" % (LISTEN_ADDR, EVENTS_PORT))
        while self.is_running:
            while self.in_stream.poll():
                resp = {}
                try:
                    msg = self.in_stream.recv_json()
                    op = msg.get('op')
                    seq = msg.get('seq')
                    if op == MAKE_PREDICTION:
                        # make this load and keep a model in memory at all times once we can afford to do that
                        resp = self.make_prediction(msg)
                    elif op == TRAIN:
                        resp = self.train(msg)
                    elif op == GET_PARAM_LIST:
                        resp = {"paramlist": classifiers.param_table}
                    elif op == GET_MODEL_LIST:
                        resp = {"classlist": classifiers.model_table.keys()}
                    elif op == GEN_FEATURES:
                        resp = self.gen_features(msg)
                    elif op == GET_STATUS_UPDATE:
                        pass
                    elif op == ANALYZE_FILE:
                        resp = self.analyze_file(msg)
                    elif op == GEN_PROJECT: # Generate a project
                        resp = self.generate_project(msg)
                    elif op == PARSE_TARGETS: # Parse targets
                        resp = parse_targets(msg)
                    elif op == CREATE_SCRIPT: # Create a script
                        resp = self.create_script(msg)
                    elif op == IMPORT_MODEL: # Import a model from s3 or any other source to the local fs
                        resp = self.import_model(msg)
                    if not isinstance(resp, dict):
                        resp = {'data': resp}
                    resp.update({'seq': seq})
                except Exception as e:
                    resp = {'status': 'err', 'message': str(e)}
                    log.error(str(e))
                    log.error(traceback.format_exc())
                self.reply(resp)
        print("Shutting down server")


