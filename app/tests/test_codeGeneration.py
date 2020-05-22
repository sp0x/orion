from unittest import TestCase
import mock

from code_generation import generate_script, generate_feature_function, generate_features_module
from code_generation.project import Project
from ml import MlScript
from tests.test_helpers import get_targets, get_data_flags, load_temp_module, \
    read_testfile
import parso
import tests.test_helpers
from code_generation.helpers import *

client = 'test_client'
name = 'mod1'


def get_test_data():
    data = [{'temperature': 1}]
    return data


features = [
    {'type': 'direct', 'title': 'f_first', 'key': 'temperature'},
]
features_with_custom = [
    {'type': 'direct', 'title': 'f_first', 'key': 'temperature'},
    {'type': 'custom', 'title': None, 'content': read_testfile("features/feature1.py")}
]
params = {
    'features': features,
    'targets': get_targets(),
    'data_flags': get_data_flags(),
    'grouping': '',
    'client': client,
    'name': name,
    'model_id': 3,
    'use_featuregen': False
}


class TestCodeGeneration(TestCase):

    def test_generate_feature_function(self):
        src = generate_feature_function({'type': 'direct', 'title': 'f_first', 'key': 'temperature'})
        assert src is not None
        function = parso.parse(src)
        assert function is not None
        functions = function.children
        assert has_func(functions, 'f_first')

    def test_generate_features_module(self):
        data = get_test_data()
        src = generate_features_module({
            'common': [
                {'type': 'direct', 'title': 'f_first', 'key': 'temperature'}
            ],
            'pm10': [
                {'type': 'direct', 'title': 'f_second', 'key': 'temperature'},
                {'type': 'direct', 'title': 'f_third', 'key': 'temperature'}
            ]
        })
        module = parso.parse(src)
        assert module is not None
        assert has_func(module.children, 'f_first')
        assert has_func(module.children, 'f_second')
        assert has_func(module.children, 'f_third')
        src_mod, file = load_temp_module(src)
        result = src_mod.f_first(data)
        file.close()
        assert isinstance(result, list)
        assert result[0] == 1

    @mock.patch('envir.get_base_path', side_effect=tests.test_helpers.get_base_path)
    def test_generate_project_zip(self, get_base_path):
        from tests.test_helpers import get_testfile
        client = 'test_client'
        name = 'mod1'
        script_path = get_testfile('script.py')
        with open(script_path, 'r') as script_fs:
            script_code = script_fs.read()
            project = Project(client, name)
            project.write_script(script_code)
            project.move_temp_assets()
            proj_zip = project.get_zip()
            assert proj_zip is not None

    @mock.patch('envir.get_base_path', side_effect=tests.test_helpers.get_base_path)
    def test_add_features_module(self, get_base_path):
        proj = Project(client, name)
        ftr_src = ''
        proj.add_features_module(None, ftr_src)
        assert len(proj.features) == 1
        assert proj.features[0]['target'] is None
        assert proj.features[0]['src'] == ftr_src

    @mock.patch('envir.get_base_path', side_effect=tests.test_helpers.get_base_path)
    def test_generate_script(self, get_base_path):
        """

        :return:
        """
        scr = generate_script(params)
        features_src = generate_features_module({'common': features})
        proj = Project(params['client'], params['name'])
        assert scr is not None
        module = parso.parse(scr)
        assert has_class(module.children, 'ModelScript')
        clsdeff = get_class(module.children, 'ModelScript')
        fns_node = clsdeff.children[6].children
        assert has_func(fns_node, '_get_features')
        assert has_func(fns_node, 'get_targets')
        assert has_func(fns_node, 'get_grouping')
        assert has_func(fns_node, 'get_file')
        assert has_func(fns_node, 'get_path')
        assert has_func(fns_node, 'input_flags')
        proj.write_script(scr)
        proj.add_features_module(None, features_src)
        src_mod, file = load_temp_module(scr)
        file.close()

    @mock.patch('envir.get_base_path', side_effect=tests.test_helpers.get_base_path)
    def test_feature_extraction(self, base_path):
        data = get_test_data()
        proj = Project(params['client'], params['name'])
        script = proj.create_script(params['features'], params['targets'], get_data_flags(), '', params['model_id'],
                                    params['use_featuregen'])
        assert script is not None
        assert isinstance(script, MlScript)
        feature_vals = script.get_features(data)
        assert feature_vals is not None
        assert 'f_first' in feature_vals
        assert feature_vals['f_first'][0] == 1

    @mock.patch('envir.get_base_path', side_effect=tests.test_helpers.get_base_path)
    def test_custom_feature_usage(self, get_base_path):
        import types
        data = get_test_data()
        proj = Project(params['client'], params['name'])
        features = features_with_custom
        script = proj.create_script(features, params['targets'], get_data_flags(), '', params['model_id'],
                                    params['use_featuregen'])
        assert script is not None
        assert isinstance(script, MlScript)
        feature_vals = script.get_features(data)
        assert feature_vals is not None
        assert 'f_first' in feature_vals
        assert feature_vals['f_first'][0] == 1
        assert isinstance(feature_vals['calc'], types.GeneratorType)
