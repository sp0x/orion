import unittest
from ml.neural_networks import NNBuilder
from tests.test_helpers import extract_keras_connections
from collections import OrderedDict

class NeuralNetworkTest(unittest.TestCase):

    def test_simple_queueing(self):
        expected_q = [1,2,3,4,5]
        nodes = range(1,5)
        conns = {
            1:{'out':[2,3]},
            2:{'out':[4]},
            3:{'out':[4]},
            4:{'out':[5]},
            5:{'out':[]}
        }
        frontier = [1]
        q = NNBuilder.__prepare_build_q__(conns, nodes, frontier)
        self.assertListEqual(expected_q, list(q))

    def test_complex_queueing(self):
        expected_q = [1,2,4,5,6,7,3,8,9]
        nodes = range(1,10)
        conns = {
            1:{'out':[3]},
            2:{'out':[4,5]},
            3:{'out':[9]},
            4:{'out':[6]},
            5:{'out':[7]},
            6:{'out':[8]},
            7:{'out':[8]},
            8:{'out':[9]},
            9:{'out':[]}
        }
        frontier=[1,2]
        q = NNBuilder.__prepare_build_q__(conns, nodes, frontier)
        self.assertListEqual(expected_q, list(q))

    def test_build_simple_neural_network_from_json(self):
        json = {
            'layers':[
                {
                    'name':'l1',
                    'type':'Input',
                    'activation': 'relu',
                    'neurons':10,
                },
                {
                    'name': 'l2',
                    'type': 'Dense',
                    'neurons': 20,
                    'activation':'relu'
                },
                {
                    'name': 'l3',
                    'type': 'Dense',
                    'neurons': 1,
                    'activation': 'relu'
                },
            ],
            'connections':{
                'l1':{'in':[],'out':['l2']},
                'l2':{'in':['l1'],'out':['l3']},
                'l3':{'in':['l2'],'out':[]}
            }
        }
        nnb = NNBuilder(json)
        n = nnb.build_rnn()
        layers = n.get_config()
        gen_cons = extract_keras_connections(layers['layers'])
        design_cons = OrderedDict(sorted(json['connections'].items()))
        self.assertDictEqual(design_cons, OrderedDict(sorted(gen_cons.items())))

    def build_complex_neural_network_from_jon(self):
        json = []

    def passing_unconnected_layer(self):
        pass
