import os

from ml import get_pipeline_latest
from ml.feature_functions import gen_feature_expr
from ml.features import get_feature_name
from utils import file_exists, update_lines
from code_generation.helpers import *


def _generate_custom_feature_function(feature, prefix=''):
    # import parso
    # f_title = feature['title']
    # ftr_type = feature['type']
    # template_path = get_template_file(['features', 'custom_feature.py'])
    feature_content = feature['content']
    if feature_content is None or len(feature_content) == 0:
        raise Exception("Custom feature must have code.")
    return feature_content

    # feature_ast = parso.parse(feature_content)
    # if f_title is None or len(f_title)==0:
    #     f_title = get_first_function_name(feature_ast)
    # if f_title is None or len(f_title) == 0:
    #     raise Exception
    # # clojures ..
    # # fc_suffix = "return " + f_title
    # # feature_content = feature_content + '\n' + fc_suffix
    # #feature_content = update_lines(feature_content, lambda l: (prefix + '  ' + l))
    # return feature_content
    # # # Make the feature function return the feature cal function
    # # with open(template_path) as tpl:
    # #     template = tpl.read()
    # #     template = template.replace('{{name}}', f_title)
    # #     template = template.replace('{{content}}', feature_content)
    # #     return template


def generate_feature_function(feature, prefix=''):
    template_path = get_template_file(['features', 'features_function.py'])
    f_title = feature['title']
    ftr_type = feature['type']
    # ftr_name = feature['name'] if 'name' in feature else None
    if ftr_type == 'custom':
        return _generate_custom_feature_function(feature, prefix)
    else:
        with open(template_path) as tpl:
            template = tpl.read()
            content = 'feature = ' + gen_feature_expr(feature) + "\n"
            content += prefix + "  return feature\n"
            template = template.replace('{{name}}', f_title)
            template = template.replace('{{content}}', content)
            return template


def generate_feature_eval_function(prefix=''):
    """
    Generates a function that evals all features inside the model module
    :param prefix:
    :return:
    """
    template_path = get_template_file(['features', 'eval.py'])
    with open(template_path) as f_tpl:
        template = f_tpl.read()
        template = update_lines(template, lambda l: (prefix + l))
        return template


def get_features_module_name(target_name):
    return "feature_functions"
    # return "features_" + target_name if target_name is not None and len(target_name) > 0 else "features_common"


def generate_features_package_init(feature_dicts, prefix=''):
    """
    Gets the source of the __init__.py file for a features package.
    :param feature_dicts:
    :param prefix:
    :return:
    """
    template_path = get_template_file(['features', '__init__.py'])

    with open(template_path) as f_tpl:
        template = f_tpl.read()
        feature_setters = ""
        feature_mods_imports = ""
        # Go over specific target features
        imported_mods = []
        for tgt in [x for x in feature_dicts if x != 'common']:
            features_mod = get_features_module_name(tgt)
            ftr_name = get_feature_name(ftr)
            if features_mod not in imported_mods:
                feature_mods_imports += "import features." + features_mod + "\n"
                imported_mods.append(features_mod)
            for ftr in feature_dicts[tgt]:
                fc = prefix + '        output[\'{0}\'] = {1}.{0}(elements)'.format(ftr_name, features_mod)
                feature_setters += fc + "\n"

        # Go over common features for all targets
        for ftr in feature_dicts['common']:
            features_mod = get_features_module_name(None)
            ftr_name = get_feature_name(ftr)
            if features_mod not in imported_mods:
                feature_mods_imports += "import features." + features_mod + "\n"
                imported_mods.append(features_mod)

            fc = prefix + '        output[\'{0}\'] = {1}.{0}(elements)'.format(ftr_name, features_mod)
            feature_setters += fc + "\n"

        template = template.replace('{{features}}', feature_setters)
        template = template.replace('{{imports}}', feature_mods_imports)
        return template
        # template = update_lines(template, lambda l: (prefix + l) if '{{features}}' not in l else l)


def generate_features_module(feature_dicts, prefix=''):
    """
    Generates a module that contains all the features code.
    :param feature_dicts:
    :param prefix:
    :return:
    """
    template_path = get_template_file(['features', 'target_features.py'])
    with open(template_path) as f_tpl:
        template = f_tpl.read()
        template = update_lines(template, lambda l: (prefix + l) if '{{features}}' not in l else l)
        feature_funcs = ""

        for tgt in [x for x in feature_dicts if x != 'common']:
            for ftr in feature_dicts[tgt]:
                feature_funcs += generate_feature_function(ftr, prefix) + "\n"

        for ftr in feature_dicts['common']:
            feature_funcs += generate_feature_function(ftr, prefix) + "\n"

        template = template.replace("{{features_content}}", feature_funcs)

        return template


def get_template_file(file):
    if isinstance(file, list):
        file = os.path.join(*file)
    return os.path.join(os.path.dirname(__file__), '..', 'data', file)


def generate_grouping_content(grouping):
    if grouping is None or len(grouping) == 0:
        grouping = "{}"
    output = "    gr_config = " + str(grouping) + "\n"
    output += "    return gr_config\n"
    return output


def generate_targets_content(targets):
    output = ""
    if targets is None or len(targets) == 0:
        targets = "{}"
    output += "    tg_config = " + str(targets) + "\n"
    output += "    return tg_config\n"
    return output


def generate_input_flags(data_flags):
    output = ""
    if data_flags is None or len(data_flags) == 0:
        data_flags = "{}"
    output += "    flags = " + str(data_flags) + "\n"
    output += "    return flags"
    return output


def generate_script(params):
    """

    :param params:
    :return:
    """
    features = params["features"]
    mod_features = generate_feature_eval_function("  ")  # generate_features_module({'common': features}, "  ")
    mod_features = mod_features.replace("def get_features(elements)", "def get_features(self, elements)")
    grouping_content = generate_grouping_content(params["grouping"] if 'grouping' in params else '')
    targets_content = generate_targets_content(params["targets"])
    input_flags_cont = generate_input_flags(params['data_flags'])
    use_featuregen = params['use_featuregen']
    client = str(params["client"])
    name = str(params["name"])
    model_id = str(params["model_id"])

    template_path = get_template_file('script_module.py')
    with open(template_path) as script_tpl:
        template = script_tpl.read()
        template = template.replace("{{username}}", client)

        template = template.replace("{{name}}", name)
        template = template.replace("{{model_id}}", model_id)
        template = template.replace("{{features_function}}", mod_features)
        template = template.replace("{{grouping}}", grouping_content)
        template = template.replace("{{targets}}", targets_content)
        template = template.replace("{{input_flags}}", input_flags_cont)
        template = template.replace("{{use_featuregen}}", str(use_featuregen))
    return template


def zip_project(clfs, proj):
    from utils import zipdir
    import zipfile
    path = proj['path']
    name = proj['name']
    # We zip up the project
    project_zip = clfs.get([path, '..', name + '.zip'])
    with zipfile.ZipFile(project_zip, 'w', zipfile.ZIP_BZIP2) as zipf:
        zipdir(path, zipf)
    # We need to package these files:
    # script.py
    # model.pickl
    # pipeline.py for refitting
    return project_zip

#
# def generate_project(params):
#     """
#         Generates a new project template
#         Settings:
#             client: Client username
#             script: python code { code : '..' }
#             name: The project name
#     :param params:
#     :return:
#     """
#     import ntpath
#     from shutil import copyfile
#     from helpers import ClientFs
#     from ml import load_ml_script, load_ml_script_abs, get_model_filepath
#     from ml.storage import get_local_model_source
#
#     script = params['script']
#     script_code = script['code']
#     client = params['client']
#     name = params['name']
#     clfs = ClientFs(client)
#     project_path = clfs.get(['projects', name])
#     script_path = clfs.get([project_path, 'script.py'])
#     with open(script_path, 'w') as f_script:
#         f_script.write(script_code)
#     script_instance = load_ml_script_abs(script_path)
#     targets = script_instance.get_targets()
#     model_id = script_instance.model_id
#     # Move all models
#     for t in targets:
#         target_col = t['column']
#         model_filepath = get_model_filepath(client, model_id, target_col)
#         model_filename = ntpath.basename(model_filepath)
#         model_source = get_local_model_source(model_filepath)
#         # We copy the model to our project dir
#         model_source.copy_to(os.path.join(project_path, model_filename))
#         # Pipeline
#         pipeline_file = get_pipeline_latest('tpot', target_col, client)
#         pipeline_filename = ntpath.basename(pipeline_file)
#         dest_pipeline_file = os.path.join(project_path, pipeline_filename)
#         if file_exists(pipeline_file):
#             copyfile(pipeline_file, dest_pipeline_file)
#
#     return project_path
