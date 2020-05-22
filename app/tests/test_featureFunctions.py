from unittest import TestCase
from ml.feature_functions import direct_template, gen_feature_expr


class TestFeatureFunctions(TestCase):

    def test_direct_template(self):
        fn = direct_template(None, 'temperature')
        assert fn == 'field(elements, \'temperature\')'

    def test_gen_feature_expr(self):
        ftr = gen_feature_expr({'type': 'direct', 'title': 'f_first', 'key': 'temperature'})
        assert ftr is not None
        assert ftr == 'field(elements, \'temperature\')'
