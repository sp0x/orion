import sys
import os

from ml import MlScript

class ModelScript(MlScript):
  def __init__(self):
    self.client = '{{username}}'
    self.name = '{{name}}'
    self.model_id = {{model_id}}
    self.use_featuregen = {{use_featuregen}}

  def get_file(self):
    return __file__

  def get_path(self):
    return os.path.dirname(self.get_file())

{{features_function}}

  def get_targets(self):
{{targets}}

  def get_grouping(self, e):
{{grouping}}

  def input_flags(self):
{{input_flags}}
