#
# Automl optimizers
#
import multiprocessing
import logging
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from utils import hasmethod, reduce_dim, regression_report
import settings as settings
from constants import RANDOM_SEED
import uuid
import os
from sklearn.metrics.classification import type_of_target
from autosklearn.constants import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION, \
    REGRESSION
import numpy as np
from autosklearn.metrics import REGRESSION_METRICS, CLASSIFICATION_METRICS
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import autosklearn.regression
import pandas as pd
import traceback
import time
import pydevd

log = logging.getLogger(__name__)
TASK_REGRESSION = 'regression'
TASK_CLASSIFICATION = 'classification'


# hdlr = logging.FileHandler(app.settings.get_log_file(__name__))
# log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# hdlr.setFormatter(log_formatter)
# log.addHandler(hdlr)
# log.setLevel(logging.DEBUG)


def detect_task(y, type='none'):
    task = type_of_target(y)
    if 'continuous' in task or type == 'regression':
        return REGRESSION
    elif 'multicalss' in task or type == 'multiclass':
        return MULTICLASS_CLASSIFICATION
    elif 'multilabel' in task or type == 'multilabel':
        return MULTILABEL_CLASSIFICATION
    else:
        return BINARY_CLASSIFICATION


def get_autosklearn_scorer(name):
    if name in CLASSIFICATION_METRICS:
        return CLASSIFICATION_METRICS[name]
    else:
        return REGRESSION_METRICS.get(name, None)


def get_target_type(targets_data):
    from sklearn.metrics.classification import type_of_target
    if isinstance(targets_data, pd.DataFrame):
        if len(targets_data.columns) > 1:
            tmp = pd.DataFrame()
            tmp[targets_data.columns[0]] = targets_data[targets_data.columns[0]]
            targets_data = tmp
        target_type = type_of_target(targets_data.as_matrix().ravel())
    else:
        target_type = type_of_target(targets_data.ravel())
    return target_type


class MlBackend(BaseSearchCV):
    def __init__(self, estimator=None, scoring='accuracy'):
        from sklearn.metrics import get_scorer
        super(MlBackend, self).__init__(estimator=estimator, scoring=get_scorer(scoring))
        self.client = None
        self.klass = None
        self.target_name = None
        self.current_model = None
        self.task_type = None

    def set_task_type(self, task_type):
        assert task_type is not None, "Task type could not be none."
        # pydevd.settrace('10.10.1.5', port=55448, stdoutToServer=True, stderrToServer=True)
        self.task_type = task_type
        self.resolve_klass()
        return self

    def resolve_klass(self):
        raise Exception('Stub')

    def set_scoring(self, scoring):
        if scoring == 'auto':
            assert self.task_type is not None, "If you want to use `auto` scoring please set the task type."
            self.scoring = settings.default_scoring[self.task_type]
        else:
            self.scoring = scoring


class AutoMlSearch(MlBackend):
    def __init__(self, klass, temp, out, ftypes=None, scoring='accuracy', task_type='none'):
        super(AutoMlSearch, self).__init__(estimator=None, scoring=scoring)
        if not os.path.exists(temp):
            os.makedirs(temp)
        if not os.path.exists(out):
            os.makedirs(out)
        self.temp_folder = temp
        self.out_folder = out
        self.klass = klass
        self.best_params_ = None
        self.ftypes = ftypes
        self.task_type = task_type

    def get_task_type(self, targets_data):
        """
        Note: use this only with single column targets
        :param targets_data:
        :return:
        """
        target_type = get_target_type(targets_data.as_matrix().ravel())
        if target_type in ['multiclass', 'continious']:
            return TASK_REGRESSION
        elif target_type == 'binary':
            return TASK_CLASSIFICATION
        return TASK_CLASSIFICATION

    def get_optimizer(self, task_type):
        from autosklearn.automl import AutoMLClassifier, AutoMLClassifier
        if task_type == TASK_REGRESSION:
            return AutoMLClassifier
        elif task_type == TASK_CLASSIFICATION:
            return AutoMLClassifier
        return None

    def fit(self, X, y=None, groups=None, **fit_params):
        processes = []
        task_type = detect_task(y, self.task_type)
        data_name = str(uuid.uuid4())
        spawn_classifier = self.__get_spawn_classifier(X, y, self.klass, self.temp_folder, self.out_folder, self.ftypes)
        log.info('Spawning search instances')
        for i in range((multiprocessing.cpu_count() // 2) or 4):  # set this at roughly half of your cores
            p = multiprocessing.Process(target=spawn_classifier, args=(i, data_name))
            p.start()
            processes.append(p)
        log.info('Searching for best model')
        for p in processes:
            p.join()

        log.info('Building ensemble')
        automl = self.klass(
            time_left_for_this_task=40,
            per_run_time_limit=40,
            ml_memory_limit=1024,
            shared_mode=True,
            ensemble_size=50,
            ensemble_nbest=200,
            tmp_folder=self.temp_folder,
            output_folder=self.out_folder,
            initial_configurations_via_metalearning=0,
            seed=1,
        )
        # Both the ensemble_size and ensemble_nbest parameters can be changed now if
        # necessary
        log.info('Training final model')
        log.info(str(automl))
        automl.fit_ensemble(
            y,
            task=task_type,
            metric=get_autosklearn_scorer(self.scoring),
            precision='32',
            ensemble_size=20,
            ensemble_nbest=50,
            dataset_name=data_name,
        )
        # automl.refit(X, y)
        # self.feature_importances_ = np.std(X, 0)*automl.coef_

        weights = automl.get_models_with_weights()
        dims = len(X[0])
        npx = np.std(X, 0)
        divby = 0
        arr = []
        for w, k in weights:
            log.info("Models with Weights: {0} {1} ".format(type(k).__name__, w));
            if hasattr(k, '_final_estimator') and hasattr(k._final_estimator.choice.estimator, 'feature_importances_'):
                ft = k._final_estimator.choice.estimator.feature_importances_
                if (len(ft) < dims): ft = np.pad(ft, mode='constant', pad_width=[0, dims - len(ft)])
                ft = w * reduce_dim(ft, dims) if len(ft) > dims else ft
            elif hasattr(k, '_final_estimator') and hasattr(k._final_estimator.choice.estimator, 'coef_'):
                ft = k._final_estimator.choice.estimator.coef_
                if (len(ft) < dims): ft = np.pad(ft, mode='constant', pad_width=[0, dims - len(ft)])
                ft = w * npx * reduce_dim(ft, dims) if len(ft) > dims else ft
            else:
                continue
            divby += 1
            arr.append(ft)
        try:
            log.info("arr shape " + str(map(np.shape, arr)))
            arrsums = np.sum(arr, axis=0)
            log.info("arrsum shape" + str(arrsums.shape))
            feature_importances_ = arrsums / divby
            setattr(automl, 'feature_importances_', feature_importances_)
        except Exception as e:
            log.error(str(e))
            log.error(traceback.format_exc())

        self.best_estimator_ = automl
        if hasattr(automl, 'cv_results_'):
            self.cv_results_ = automl.cv_results_
        else:
            self.cv_results_ = []

    def sprint_statistics(self):
        if hasattr(self.best_estimator_, 'configuration_space'):
            return self.best_estimator_.sprint_statistics()
        else:
            return ''

    def predict_proba(self, X):
        if not self.is_regression():
            from autosklearn.automl import AutoMLClassifier
            if isinstance(self.best_estimator_._automl, AutoMLClassifier):  # Bug in automl 0.3.0
                return self.best_estimator_.predict(X)
            else:
                proba = self.best_estimator_.predict_proba(X)
                # Todo: update this when we have more classes
                return proba[:, 1]
        else:
            return self.best_estimator_.predict(X)

    def accuracy_score(self, test, prediction):
        if not self.is_regression():
            return accuracy_score(test, prediction)
        else:
            return self.best_estimator_.score(test, prediction)

    def report(self, y_true, y_pred):
        if not self.is_regression():
            return classification_report(y_true, y_pred)
        else:
            return regression_report(y_true, y_pred)

    def is_regression(self):
        return self.klass == autosklearn.regression.AutoSklearnRegressor

    def predict(self, x):
        prediction = self.best_estimator_.predict(x)
        prediction = np.ravel(prediction)
        return prediction

    def get_model(self):
        return self.get_models()[0]

    def get_models(self):
        return list(self.best_estimator_._automl.models_.values())

    def feature_importance(self):
        pass

    def __get_spawn_classifier(self, X_train, y_train, klass, tmp, out, ftypes=None,
                               exclude_estimators=['DummyRegressor', 'DummyClassifier']):
        def spawn_classifier(seed, dataset_name='something'):
            """Spawn a subprocess.
            auto-sklearn does not take care of spawning worker processes. This
            function, which is called several times in the main block is a new
            process which runs one instance of auto-sklearn.
            """
            # Use the initial configurations from meta-learning only in one out of
            # the four processes spawned. This prevents auto-sklearn from evaluating
            # the same configurations in four processes.
            if seed == 0:
                initial_configurations_via_metalearning = 25
                smac_scenario_args = {}
            else:
                initial_configurations_via_metalearning = 0
                smac_scenario_args = {'initial_incumbent': 'RANDOM'}
            if dataset_name != None:
                dataset_name = str(dataset_name)

            exclude = ['DummyRegressor'] if self.is_regression() else ['DummyClassifier']
            # Arguments which are different to other runs of auto-sklearn:
            # 1. all classifiers write to the same output directory
            # 2. shared_mode is set to True, this enables sharing of data between
            # models.
            # 3. all instances of the AutoSklearnClassifier must have a different seed!
            automl = klass(
                time_left_for_this_task=480,  # sec., how long should this seed fit process run
                per_run_time_limit=60,  # sec., each model may only take this long before it's killed
                ml_memory_limit=1024,  # MB, memory limit imposed on each call to a ML algorithm
                shared_mode=True,  # tmp folder will be shared between seeds
                tmp_folder=tmp,
                output_folder=out,
                delete_tmp_folder_after_terminate=False,
                delete_output_folder_after_terminate=False,
                ensemble_size=0,  # ensembles will be built when all optimization runs are finished
                initial_configurations_via_metalearning=initial_configurations_via_metalearning,
                include_preprocessors=["no_preprocessing", ],
                seed=seed,
                smac_scenario_args=smac_scenario_args,
                exclude_estimators=exclude_estimators
            )
            if ftypes:
                # dataset_name must be a string
                automl.fit(X_train, y_train, feat_type=ftypes, dataset_name=str(dataset_name))
            else:
                automl.fit(X_train, y_train)

        return spawn_classifier


class TPOTSearch(MlBackend):
    def __init__(self, scoring='accuracy'):
        super(TPOTSearch, self).__init__(estimator=None, scoring=scoring)
        # if klass is not None:
        #     self.instance = self.klass(generations=10,
        #                                scoring=self.scoring,
        #                                n_jobs=-1,
        #                                config_dict='TPOT light',
        #                                early_stop=6,
        #                                verbosity=2,
        #                                disable_update_check=True
        #                                )
        # self.client = client
        # self.target_name = target_name
        self.current_model = None
        self.instance = None

    def resolve_klass(self):
        assert self.task_type is not None
        optimizer = self.get_optimizer(self.task_type)
        self.klass = optimizer

    def get_task_type(self, targets_data, task_type=None):
        """
        Note: use this only with single column targets
        :param task_type:
        :param targets_data:
        :return:
        """
        target_type = get_target_type(targets_data)
        if target_type in ['continious']:
            if task_type is not None and task_type != TASK_REGRESSION:
                raise Exception("Bad task type {0} for data type {1}".format(task_type, target_type))
            return TASK_REGRESSION
        elif target_type in ['multiclass', 'binary']:
            if task_type is not None:
                return task_type
            else:
                return TASK_CLASSIFICATION
        return TASK_CLASSIFICATION

    def scale(self, data):
        """

        :param data:
        :return: The scaler and the scaled data
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_np = np.nan_to_num(data)
        data_scaled = scaler.fit_transform(data_np)
        return scaler, data_scaled

    def scale_target(self, targets_data):
        """

        :param targets_data
        :return: The scaler and the scaled data
        """
        assert self.task_type is not None or len(self.task_type) == 0, 'Task type is not set.'
        data_np = np.nan_to_num(targets_data)
        if self.task_type == TASK_REGRESSION:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data_np)
            return scaler, data_scaled
        else:
            return None, data_np

    def get_optimizer(self, task_type):
        from tpot import TPOTRegressor, TPOTClassifier
        if task_type == TASK_REGRESSION:
            return TPOTRegressor
        elif task_type == TASK_CLASSIFICATION:
            return TPOTClassifier
        return None

    def get_instance(self, reset=False):
        if self.instance is not None and not reset:
            return self.instance
        if self.klass is None:
            self.resolve_klass()

        self.instance = self.klass(generations=10,
                                   scoring=self.scoring,
                                   n_jobs=-1,
                                   config_dict='TPOT light',
                                   early_stop=4,
                                   verbosity=2,
                                   disable_update_check=True
                                   )
        return self.instance

    def fit(self, X, y=None, pipeline_datasize=500, groups=None, **fit_params):
        import utils
        from ml import get_pipeline_filepath
        assert X.shape[0] > pipeline_datasize, "X[0]({0}) has less or equal elements compared to the pipeline size" \
                                               "({1}).".format(X.shape[0], pipeline_datasize)
        assert y.shape[0] > pipeline_datasize, "y[0]({0}) has less or equal elements compared to the pipeline size" \
                                               "({1}).".format(y.shape[0], pipeline_datasize)
        instance = self.get_instance(False)
        print("X shape: " + str(X.shape))
        print("y shape: " + str(y.shape))
        X_tpot_pipeline = X[:pipeline_datasize, :]
        y_tpot_pipeline = y[:pipeline_datasize, :]
        # X = X[pipeline_datasize:, :]
        # y = y[pipeline_datasize:, :]
        print("Picking pipeline: {0} target {1} ({2} - {3})".format(str(X_tpot_pipeline.shape),
                                                                    str(y_tpot_pipeline.shape), self.task_type,
                                                                    self.scoring))
        instance.fit(X_tpot_pipeline, y_tpot_pipeline)
        output_filename = get_pipeline_filepath('tpot', self.target_name, self.client)
        self.export_src(output_filename)
        self.__format_pipeline(output_filename)
        pipeline_mod = self.__get_pipeline(output_filename)
        # We built the pipeline, let's fit it now
        df = pd.DataFrame(data=X)
        df['target'] = y
        print("Fitting pipeline: {0} {1}".format(str(df.shape), X.shape))
        trained_pipeline_results = pipeline_mod.fit(df)
        trained_pipeline = trained_pipeline_results[0]
        # print("Fitted : ")
        # print(df.head())
        self.current_model = trained_pipeline
        return {
            'pipeline': trained_pipeline,
            'module': pipeline_mod,
            'file': output_filename
        }

    def __get_pipeline(self, filename):
        import importlib.util
        spec = importlib.util.spec_from_file_location(".", filename)
        pipeline_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_mod)
        return pipeline_mod

    def is_regression(self):
        from tpot import TPOTRegressor, TPOTClassifier
        return self.klass == TPOTRegressor  # may not be but use built in scorer just in case

    def accuracy_score(self, test, prediction):
        if not self.is_regression():
            return self.current_model.score(test, prediction)
        else:
            return self.current_model.score(test, prediction)

    def report(self, y_true, y_pred):
        if not self.is_regression():
            return classification_report(y_true, y_pred)
        else:
            return regression_report(y_true, y_pred)

    def predict(self, X):
        return self.instance.predict(X)
        # return self.current_model.predict(X)

    def predict_proba(self, X):
        try:
            p = self.current_model.predict_proba(X)
            # Todo: update this when we have more classes
            return p[:, 1]
        except:
            return self.current_model.predict(X)

    def export_src(self, filename):
        return self.instance.export(filename)

    def sprint_statistics(self):
        root_model = get_model_obj(self.current_model)
        if hasattr(root_model, 'configuration_space'):
            return root_model.sprint_statistics()
        else:
            return ''

    def feature_importance(self):
        root_model = get_model_obj(self.current_model)
        if hasattr(root_model, "feature_importances_"):
            importances = root_model.feature_importances_
            return importances
        elif hasattr(root_model, "coef_"):
            importances = root_model.coef_
            return importances

    def __format_pipeline(self, f_pipeline):
        with open(f_pipeline) as f:
            s_pipeline = f.read()
        fndef = 'def fit(tpot_data):'
        s_pipeline = s_pipeline.replace(
            "tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)",
            "%s" % fndef)
        s_lines = s_pipeline.split('\n')
        buff = ''
        met_fun = False
        for line in s_lines:
            if line.find(fndef) == -1 and met_fun:
                line = '  ' + line
            elif not met_fun and line.find('%s' % fndef) != -1:
                met_fun = True
            buff += line + '\n'
        buff += '  return exported_pipeline, results'
        with open(f_pipeline, 'w') as f:
            f.write(buff)


def task_type_is_valid(targets_df, task_type):
    if task_type is None or len(task_type) == 0:
        return False
    target_type = get_target_type(targets_df)
    if target_type in ['continious']:  # Continious data must be a regression
        if task_type is not None and task_type != TASK_REGRESSION:
            return False
    return True


def get_ml_backend(targets_df, task_type=None, scoring=None):
    """
    Gets a ml backend configured with the type of target data
    :param scoring:
    :param task_type:
    :param targets_df:
    :return:
    """
    from settings import AUTOML_BACKEND
    backend = None
    if AUTOML_BACKEND == "autosklearn":
        backend = AutoMlSearch
    elif AUTOML_BACKEND == "tpot":
        backend = TPOTSearch
    if backend is None:
        return None
    instance = backend()
    if targets_df is None:
        return instance
    # pydevd.settrace('10.10.1.5', port=55448, stdoutToServer=True, stderrToServer=True)
    task_type = task_type if task_type_is_valid(targets_df, task_type) else instance.get_task_type(targets_df,
                                                                                                   task_type=task_type)
    instance.set_task_type(task_type)
    if scoring is not None:
        instance.set_scoring(scoring)
    return instance


def get_model_obj(model_info):
    if isinstance(model_info, Pipeline):
        root_model = model_info.steps[-1][1]  # The last step is always the regressor/classifier.
    elif hasattr(model_info, 'model') and isinstance(model_info['model'], Pipeline):
        root_model = model_info['model'].steps[-1][1]
    elif isinstance(model_info, dict) and 'model' in model_info:
        mod = model_info['model']
        root_model = get_model_obj(mod)
    elif not hasattr(model_info, 'model'):
        return model_info
    else:
        root_model = model_info['model']
    return root_model
