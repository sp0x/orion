#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model #save_model, load_model, Model
from keras.layers import merge
from copy import deepcopy
import collections

def get_hidden_neurons_count(Ni, Ns, No=1, a=2):
    return Ns / (a * (Ni + No))
layer_list = {}
# json descriptions:
# {
#   layers:[
#    {
#       Name:'l1' - name of the layer used to figure out what to connect to what
#       Type:'Dense' - type of the layer and name of the class in keras used to build the thing
#       activation: 'sigmoid' - string name of the activation function
#       neurons: 12 or 'auto' - int number of neurons in the layer or string 'auto' auto determine amount of neurons
#       shape: [x,y,z]        - array represents the shape of the layer
#       image_dim:[w,h,c] - array of 3 ints [width,height,channels] of an image
#                           available only for conv layers replaces neurons key
#   }]
#   connections:{
#       l1:{ - use layer name as key
#           in:[] - list of the names of layers which feed input to this (only input layers can have empty list)
#           out:[] - list of the names of layers which leech data from this (only output layers can have empty list)
#       }
#   compile:{
#     optimizer:'', - training algo
#     loss:'',      - loss function to minimize
#     metrics:['']  - metrics to use to measure accuracy
#     }
#   }
class NNBuilder:
    def __init__(self, instructions, input_size=None, output_size=None):
        self.blueprint = instructions
        self.input_size = input_size
        self.output_size = output_size

    def __get_ios__(self, io='in'):
        ios = []
        for l in self.blueprint['connections']:
            if len(self.blueprint['connections'][l][io]) == 0:
                ios.append(l)
        return ios

    @staticmethod
    def height(node, conns, visited, height_matrix):
        my_cons = conns.get(node, None)
        if not my_cons.get('out') or node in visited:
            h = height_matrix.get(node, 0)
            height_matrix[node] = h
            return h
        visited.add(node)
        h = max([NNBuilder.height(c, conns, visited, height_matrix) for c in my_cons.get('out')]) + 1
        height_matrix[node] = h
        return h

    @staticmethod
    def __prepare_build_q__(conns, nodes, frontier):
        q = collections.deque(frontier)
        visited = set(frontier)
        nodes = set(nodes)
        hm = dict()
        hs = [(c, NNBuilder.height(c, conns, set(), hm)) for c in frontier]
        while visited != nodes:
            if len(frontier) > 1:
                cur = max(hs, key=lambda x:x[1])[0]
            else:
                cur = frontier[0]
            frontier.remove(cur)
            visited.add(cur)
            cns = filter(lambda x: x not in visited and x not in q, conns[cur].get('out', []))
            if cns:
                q.extend(cns)
                frontier.extend(cns)
            hs = [(k, hm[k]) for k in frontier]
        return q

    @staticmethod
    def __get_layer__(ltype):
        module = __import__(layer_list[ltype], fromlist=[ltype])
        return getattr(module, ltype)

    @staticmethod
    def __is_visual__(ltype):
        return 'conv' in ltype.lower() or 'crop' in ltype.lower()

    def __format_blueprint__(self):
        return dict(map(lambda x:(x['name'], x), self.blueprint['layers']))

    def __build_layers__(self, q ,layer_table):
        """
        :param q: deque containing the order of layer gen
        :param layer_table: dict containing all currently initialized layers
        :return: returns dict of keras.layers instances and names
        """
        blueprint = self.__format_blueprint__()
        conns = self.blueprint['connections']
        while q:
            l = q.popleft()                                             #layer name
            i = blueprint[l]                                            #layer build instructions
            cl = NNBuilder.__get_layer__(i["type"])                     #layer type class
            if self.__is_visual__(i['type']):
                li = cl(input_shape=i['image_dim'], name=i['name'], activation=i['activation'])
            elif 'input' in i.get('type', '').lower():
                shape = tuple(i.get('shape', [i.get('neurons'),]))
                li = cl(shape=shape, name=i['name'])
            else:
                li = cl(i['neurons'], name=i['name'], activation=i['activation'])   #layer instance
            inputs = conns[l]['in']
            if len(inputs) > 0 :
                if len(inputs) > 1:
                    x = merge(map(lambda x:layer_table[x], inputs))
                else:
                    x = layer_table[inputs[0]]
                li = li(x)
            layer_table[l] = li
        return layer_table


    def build_rnn(self):
        nodes =  set([n['name'] for n in self.blueprint['layers']])
        ins = self.__get_ios__()
        outs = self.__get_ios__('out')
        q = NNBuilder.__prepare_build_q__(self.blueprint['connections'], nodes, deepcopy(ins))
        lt = self.__build_layers__(q, dict())
        model = Model(inputs=map(lambda x: lt[x], ins), outputs=map(lambda y: lt[y], outs))
        ci = self.blueprint.get('compile', dict(optimizer='rmsprop',
                                               loss='binary_crossentropy',
                                               metrics=['accuracy']))
        model.compile(optimizer=ci['optimizer'], loss=ci['loss'], metrics=ci['metrics'])
        return model
