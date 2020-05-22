import logging
from constants import RANDOM_SEED, MONITOR_RESPONSE_TIME
import numpy as np
import time, datetime
import matplotlib
from uuid import uuid4

matplotlib.use('Agg')
import uuid
from ml.feature_extraction import FeaturesWrapper
from db.data_handlers import MongoBufferedReader
from ml.targets import TargetPicker
from helpers import *
from db.encoding import EncodingHelper
from ml.experiments import Experiment, ScriptedExperiment

np.random.seed(RANDOM_SEED)
model_table = {}  # {model_name:path}
param_table = {}

# real time monitoring - gather data and build training set
# classifiers train and eval
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def build_scripted(params):
    """

    :param params:
    :return:
    """
    from db.encoding import EncodingHelper
    models = params.get('models')  # models:{model_name:params}
    data_options = params.get('options')
    client = params.get('client')
    task_type = params.get('task', None)
    script = params.get('script')
    experiment_name = params.get('experiment_name', None)  # So that we can track experiment assets
    assert experiment_name is not None
    experiment = ScriptedExperiment([], client, experiment_name)
    experiment.set_script(script)
    experiment.script_instance = script_inst = experiment.get_script_instance()
    experiment.data_source = {
        'db': data_options.get('db')
    }
    ftypes = None
    scoring = data_options['scoring']
    model_id = params.get('model_id')

    exp_models_info = get_experiment_models(model_id, models, scoring)
    f_types = exp_models_info['ftypes']
    classifiers = exp_models_info['classifiers']
    experiment.models = classifiers
    experiment.ftypes = f_types
    experiment.encoder = EncodingHelper(experiment.script_instance.input_flags())
    experiment.task_ids = params['tasks']
    experiment.is_automl = all(map(lambda x: x['type'] == 'auto', classifiers))

    return experiment


def build(params):
    """

    :param params:
    :return:
    """
    models = params.get('models')  # models:{model_name:params}
    data_options = params.get('options')
    client = params.get('client')
    task_type = params.get('task', None)
    experiment_name = params.get('experiment_name', None)  # So that we can track experiment assets
    assert experiment_name is not None
    classifiers = []
    # Extract targets
    experiment_input = get_experiment_input(params, params.get('targets'))
    data = experiment_input['data']
    data_targets = experiment_input['data_targets']

    features_wrapper = FeaturesWrapper(features=data.columns)
    features_wrapper.add_collection(experiment_input['source'], add_features=False)
    features_wrapper.drop(data_targets.columns)
    features_wrapper_dict = {t: features_wrapper for t in data_targets.columns}
    scoring = data_options['scoring']
    model_id = params.get('model_id')

    exp_models_info = get_experiment_models(model_id, models, scoring)
    f_types = exp_models_info['ftypes']
    classifiers = exp_models_info['classifiers']
    # data.to_csv('fit_input_data_{0}.csv'.format(str(targets_data.columns).replace(',', '_')))
    e = Experiment(data,
                   data_targets,
                   classifiers,
                   client=client,
                   ftypes=f_types,
                   features=features_wrapper_dict,
                   task_type=task_type,
                   experiment_name=experiment_name)
    e.encoder = EncodingHelper(data_options)
    e.task_ids = params['tasks']
    return e


def get_experiment_models(model_id, models, scoring):
    outp = {'classifiers': [], 'ftypes': None}
    classifiers = []
    # ftypes = []
    for m in models:
        # ftypes = data.dtypes
        m_params = models[m]
        # Get the model
        if m == "auto":
            cls = model = 'auto'  # no need to resolve this right now..
            # backend = get_ml_backend(data_targets)
            # cls = backend.klass
            # ftypes = list(map(lambda x: 'Categorical' if str(x) != 'float64' else 'Numerical', ftypes))
        else:
            module = __import__(model_table[m], fromlist=[m])
            klass = getattr(module, m)
            cls = klass()
        logging.info("Using optimizer: " + str(cls))
        classifiers.append(dict(model=cls,
                                params=m_params,
                                scoring=scoring,
                                type=m,
                                id=str(uuid4())[-12:],
                                model_id=model_id))
    outp['classifiers'] = classifiers
    # outp['ftypes'] = ftypes
    return outp


def train(e):
    try:
        data = e.load_data()
        logging.info("experiments for {1} started at {0}".format(str(datetime.datetime.now()), e.client))
        output = e.create_and_train()
        logging.info("experiments for {1} ended at {0}".format(str(datetime.datetime.now()), e.client))
    except Exception as ex:
        raise ex
    return output


def get_experiment_input(params, targets):
    """
    Gets input for the experiment
    :param params:
    :param targets:
    :param ml_script:
    :return:
    """
    import ml.cleanup
    import ml.preprocessing
    data_options = params.get('options')
    db_source = data_options.get('db')
    mr = MongoBufferedReader()
    data = mr.read_as_pandas_frame(db_source,
                                   data_options.get('start'),
                                   data_options.get('end'),
                                   None, True, True,
                                   fields=data_options.get('fields'))
    data = ml.cleanup.clean_data_for_training(data, data_options['fields'])
    data = ml.preprocessing.preprocess_training_data(data, data_options['fields'])
    mr.close()
    inp_targets = pd.DataFrame()
    targets_data = TargetPicker(targets, db_source).get_frame(data)
    # Assert we don`t have targets in our data
    for t in inp_targets:
        assert t['column'] not in data
    return {'data': data, 'source': db_source, 'data_targets': targets_data}
