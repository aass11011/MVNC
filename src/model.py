import  tensorflow as tf
import numpy as np
class Model(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {'name','logging'}
        for kwarg in kwargs.keys():
            assert  kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=self.name)
        self.vars = {var.name: var for var in variables}
    def fit(self):
        pass
    def predict(self):
        pass

def gaussian_noise_layer(input_layer,std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std,dtype=tf.float32)
    return input_layer + noise



class ARGA(Model):
    def __init__(self,placeholders, numView, num_features, num_clusters, **kwargs):
        super(ARGA, self).__init__(**kwargs)

        self.input = placeholders['features']
        self.num_features = num_features
        self.adjs = placeholders['adjs']
        self.dropout = placeholders['dropout']
        self.attn_drop = placeholders['attn_drop']
        self.ffd_drop = placeholders['ffd_drop']
        self.num_clusters = num_clusters
        self.numView = numView
        self.build()
    def _build(self):
        with tf.variable_scope('Encoder',reuse=None):
            self.embedding = []
            for v in range(self.numView):
                self.hidden1 = GraphConvolution(input_dim=self.num_features,output_dim=FLAGS.hidden1,adj)
                self.noise = gaussian_noise_layer(self.hidden1,0.1)
                self.embedding = GraphConvolution()

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs= self._call(inputs)
            return outputs

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class GraphConvolution(Layer):
    def __init__(self,input_dim,output_dim,adj,dropout=0.,act=tf.nn.relu,**kwargs):
        super(GraphConvolution,self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim,output_dim,name="weight")
            print('self.vars',self.vars['weights'])
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.matmul(x,self.vars['weight'])
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs