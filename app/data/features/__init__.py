#features module file
import os
import importlib
pyfile_extes = ['.py', ]
dirn = os.path.dirname(__file__)
{{imports}}
#print("Importing features from: " + dirn)
dir_files = [os.path.splitext(i)[0] for i in os.listdir(dirn) if os.path.splitext(i)[1] in pyfile_extes]
__all__ = [importlib.import_module('.%s' % filename, __package__) for filename in dir_files if not filename.startswith('__')]
print("Loaded feature modules:")
print(__all__)
del os, importlib, pyfile_extes

#To use the features, call . features_<target_name|common>.<feature_name>(elements)

class FeaturesModule(object):

    def __init__(self):
        pass

    def get_features(self, elements):
        output = dict()
{{features}}
        return output
