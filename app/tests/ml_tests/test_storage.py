from unittest import TestCase
from ml.storage import get_model_source
from ..test_helpers import get_testfile
from ml import Model


class TestMlStorage(TestCase):

    def test_fs(self):
        model_file = get_testfile("example_model.pkl")
        model_source = get_model_source("fs", model_file)
        assert model_source is not None
        loaded = model_source.load()
        assert loaded is not None

    def test_s3(self):
        # model_file = "example_model.pkl"
        model_file = "s3://netlyt.com/example_model.pkl"
        model_source = get_model_source("s3", model_file, True)
        assert model_source is not None
        loaded = model_source.load()
        assert loaded is not None

    # def test_import_from_s3(self):
    #     from ml.storage import import_model
    #     model_file = "s3://netlyt.com/example_model.pkl"
    #     model_source = get_model_source("s3", model_file, True)
    #     assert model_source is not None
    #     loaded = model_source.load()
    #     assert loaded is not None

    def test_fs_copy(self):
        model_file = get_testfile("example_model.pkl")
        model_file_dest = get_testfile("example_model_tmp.pkl")
        model_source = get_model_source("fs", model_file)
        model_dest = get_model_source("fs", model_file_dest)
        # Assert that we can copy it and open it correctly
        assert model_source.copy_to(model_dest)
        copied_model = Model(model_dest)
        assert model_dest.delete()

    def test_s3_to_fs_copy(self):
        model_file = "s3://netlyt.com/example_model.pkl"
        model_file_dest = get_testfile("example_model_tmp.pkl")
        model_source = get_model_source("s3", model_file, True)
        model_dest = get_model_source("fs", model_file_dest)
        # Assert that we can copy it and open it correctly
        assert model_source.copy_to(model_dest)
        copied_model = Model(model_dest)
        assert model_dest.delete()

    def test_fs_to_s3_copy_authed(self):
        import settings
        import boto3
        import os
        model_file_dest = "s3://data.ml.netlyt.com/example_model.pkl"
        model_file = get_testfile("example_model.pkl")

        model_source = get_model_source("fs", model_file)
        model_dest = get_model_source("s3", model_file_dest)
        # Assert that we can copy it and open it correctly
        assert model_source.copy_to(model_dest)
        copied_model = Model(model_dest)
        assert model_dest.delete()

    def test_model_loading(self):
        model_file = get_testfile("example_model.pkl")
        model_source = get_model_source("fs", model_file)
        mod = Model(model_source)
        assert hasattr(mod, '_model')
        assert hasattr(mod, 'target')
        assert mod._model is not None
        assert mod.target is not None
