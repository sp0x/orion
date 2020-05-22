import os

from envir import add_path
from helpers import ClientFs

DIR_EXP = 'exp'


class Model(object):
    """
    A model wrapper
    """
    from ml.storage import ModelSource

    def __init__(self, source: ModelSource):
        """

        :param source: The source from which to load the model.
        """
        self.source = source
        self._model = source.load()
        if 'model' not in self._model:
            raise Exception('`model` not present in model source')
        if 'columns' not in self._model:
            raise Exception('`columns` not present in model source')
        if 'target' not in self._model:
            raise Exception('`target` not present in model source')
        self.target = self._model['target']


class MlScript:
    """
    A template for model scripts.
    """

    def __init__(self):
        self.client = None
        self.name = None
        self.model_id = None
        self.use_featuregen = False

    def get_targets(self):
        """

        :return: A list of dicts for each target
        { column: TargetName, scoring: The scorer that's used, task_type: Regression/Classification/.., constraints: any  target constraints }
        """
        return []

    def get_grouping(self):
        return {}

    def input_flags(self):
        return {}

    def get_path(self):
        return os.path.dirname(__file__)

    def get_features(self, elements):
        """
        :param elements: list Features/fields that are used in the models.
        :return:
        """
        with add_path(self.get_path()):
            features = self._get_features(elements)
        return features

    def _get_features(self, elements):
        return {}

# region Script tools

def save_ml_script(client, name, script):
    client_fs = ClientFs(client)
    client_fs.save([DIR_EXP, name, 'script.py'], script['code'])


def load_ml_script_abs(script_path, *args, **kwargs) -> MlScript:
    """
    Loads a ml script from an absolute path.
    :param script_path:
    :param args:
    :param kwargs:
    :return:
    """
    import importlib.util
    import ntpath
    dir_path = os.path.dirname(script_path)
    dir_name = ntpath.basename(dir_path)
    spec = importlib.util.spec_from_file_location(dir_name, script_path)
    script_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_mod)
    script_type = script_mod.ModelScript
    instance = script_type(*args, **kwargs)
    return instance


def load_ml_script(client, exp_name, *args, **kwargs) -> MlScript:
    """
    Loads a ml script for a client for a specific experiment.

    :param client:
    :param exp_name:
    :param args:
    :param kwargs:
    :return: MlScript
    """
    client_fs = ClientFs(client)
    script_path = client_fs.get([DIR_EXP, exp_name, 'script.py'])
    instance = load_ml_script_abs(script_path, *args, **kwargs)
    return instance


# endregion

# region Pipelines
def get_pipeline_filepath(fn, name, client, usets=True):
    import time
    client_fs = ClientFs(client)
    ts = str(int(time.time())) if usets else ''
    fname = "p_{0}_{1}{2}.py".format(fn, name, ts)
    filepath = client_fs.get(['ml_data', fname])
    return filepath


def get_pipeline_latest(fn, name, client):
    import glob, os
    fname = get_pipeline_filepath(fn, name, client, usets=False)
    fname = fname[:len(fname) - 3]
    fname += '*.py'
    matching_files = glob.glob(fname)
    if len(matching_files)==0:
        return None
    latest_file = max(matching_files, key=os.path.getctime)
    return latest_file


# endregion

# region Model fs tools

def get_model_filename(model_id, target):
    """
    Gets the filename of a model
    :param model_id:
    :param target:
    :return:
    """
    filename = "model_{0}_{1}.pickle".format(model_id, target)
    return filename


def get_model_filepath(client, model_id, target):
    client_fs = ClientFs(client)
    model_fname = get_model_filename(model_id, target)
    path = client_fs.get(['models', model_fname])
    return path


def load_model(client, model_id, target):
    client_fs = ClientFs(client)
    fpath = get_model_filepath(client, model_id, target)
    mlmodel = client_fs.load_pickle(fpath)
    return mlmodel


def store_model(model, client, target):
    client_fs = ClientFs(client)
    m_id = model['model_id']
    filepath = get_model_filepath(client, m_id, target)
    return client_fs.save_pickle(filepath, model, is_abs=True)


def name_from_model(model_id):
    return "Model{0}".format(model_id)


def load_targets_models(model, targets, client):
    for t in targets:
        mmlodel = load_model(client, model.Id, t)
        yield t, mmlodel
# endregion
