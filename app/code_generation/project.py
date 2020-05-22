import os
import sys

from code_generation import get_template_file, generate_script, generate_features_module, \
    generate_features_package_init, get_features_module_name
from envir import add_path
from ml import load_ml_script_abs, get_model_filepath, get_pipeline_latest, get_model_filename
from ml.storage import get_local_model_source
from utils import load_module


class Project(object):

    def __init__(self, client, name):
        from helpers import ClientFs
        self.client = client
        self.name = name
        self.clfs = ClientFs(client)
        project_path = self.clfs.get(['projects', name])
        self.path = project_path
        self.script = self.clfs.get([project_path, 'script.py'])
        self.features = []
        self._script_instance = None
        self.features_dir = self.clfs.get([self.path, 'features'])
        self.__bootstrap()

    def __bootstrap(self):
        """
        Creates the initial assets required for each project.
        :return:
        """
        from shutil import copyfile
        # features_template_path = get_template_file(['features', '__init__.py'])
        # features_init_file = os.path.join(self.features_dir, '__init__.py')
        # copyfile(features_template_path, features_init_file)

    def write_script(self, code):
        with open(self.script, 'w') as f_script:
            f_script.write(code)

    def create_script(self, features, targets, data_flags, grouping, model_id, generateFeatures):

        """
        Creates a script for the project and returns an instance of it.
        :param features:
        :param targets:
        :param data_flags:
        :param grouping:
        :param model_id:
        :param generateFeatures:
        :return:
        """
        script_src = generate_script({
            'features': features,
            'targets': targets,
            'data_flags': data_flags,
            'grouping': grouping,
            'client': self.client,
            'name': self.name,
            'model_id': model_id,
            'use_featuregen': generateFeatures
        })
        self.write_script(script_src)
        if isinstance(features, list):
            features = {'common': features}
        features_src = generate_features_module(features)
        features_package = generate_features_package_init(features)
        self.add_features_module(None, features_src)
        self._set_features_package_init(features_package)
        with add_path(self.path):
            instance = self.get_script_instance()
        return instance


    def get_script_instance(self):
        """
        Gets an instance of the script. It's created only once.
        :return:
        """
        if self._script_instance is not None:
            return self._script_instance
        script_instance = load_ml_script_abs(self.script)
        self._script_instance = script_instance
        return script_instance

    def get_model(self, target_col):
        script_instance = self.get_script_instance()
        model_id = script_instance.model_id
        model_filepath = get_model_filepath(self.client, model_id, target_col)
        model_source = get_local_model_source(model_filepath)
        return model_source, model_filepath

    def get_model_path(self, target):
        """

        :param target:
        :return:
        """
        script = self.get_script_instance()
        model_id = script.model_id
        model_fname = get_model_filename(model_id, target)
        model_filepath = self.clfs.get([self.path, model_fname])
        return model_filepath

    def move_temp_assets(self):
        """
        Moves any temporary assets to the project directory
        :return:
        """
        import ntpath
        from shutil import copyfile
        import os
        from utils import file_exists
        script = self.get_script_instance()
        targets = script.get_targets()
        model_id = script.model_id
        client = self.client
        project_path = self.path
        for t in targets:
            target_col = t['column']
            model_filepath = get_model_filepath(client, model_id, target_col)
            model_filename = ntpath.basename(model_filepath)
            model_source = get_local_model_source(model_filepath)
            dest_model_path = os.path.join(project_path, model_filename)
            if model_source.exists():
                # We copy the model to our project dir
                model_source.copy_to(dest_model_path)
            # Pipeline
            pipeline_file = get_pipeline_latest('tpot', target_col, client)
            if pipeline_file is not None:
                pipeline_filename = ntpath.basename(pipeline_file)
                dest_pipeline_file = os.path.join(project_path, pipeline_filename)
                if file_exists(pipeline_file):
                    copyfile(pipeline_file, dest_pipeline_file)

    def get_zip(self):
        from utils import zipdir
        import zipfile
        path = self.path
        name = self.name
        # We zip up the project
        project_zip = self.clfs.get([path, '..', name + '.zip'])
        with zipfile.ZipFile(project_zip, 'w', zipfile.ZIP_BZIP2) as zipf:
            zipdir(path, zipf)
        # We need to package these files:
        # script.py
        # model.pickl
        # pipeline.py for refitting
        return project_zip

    def _set_features_package_init(self, source):
        features_file_path = os.path.join(self.features_dir, '__init__.py')
        with open(features_file_path, 'wb') as features_file:
            features_file.write(source.encode('utf-8'))
        return self

    def add_features_module(self, target, feature_mod_src):
        """

        :param target:
        :param feature_mod_src:
        :return:
        """
        fname = get_features_module_name(target)
        self.features.append({
            'target': target,
            'src': feature_mod_src
        })
        features_file_path = os.path.join(self.features_dir, fname + '.py')
        with open(features_file_path, 'wb') as features_file:
            features_file.write(feature_mod_src.encode('utf-8'))
        return self
