import  tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('input_view', 1, 'View No. informative view, ACM:0, DBLP:1')
flags.DEFINE_float('weight_decay', 0.0001, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('fea_decay', 0.5, 'feature decay.')
flags.DEFINE_float('weight_R', 0.001, 'Weight for R loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('attn_drop', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('ffd_drop', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 50, 'number of iterations.')
flags.DEFINE_integer('n_clusters', 3, 'predict label early stop.')
flags.DEFINE_float('kl_decay', 0.1, 'kl loss decay.')


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

        self.inputs = placeholders['features']
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
            self.embeddings = []
            for v in range(self.numView):
                self.hidden1 = GraphConvolution(input_dim=self.num_features,output_dim=FLAGS.hidden1,adj=self.adjs[v],act=tf.nn.relu,dropout=self.dropout,
                                                logging=self.logging,name='e_dense_1_'+str(v))(self.inputs)
                self.noise = gaussian_noise_layer(self.hidden1,0.1)
                embeddings = GraphConvolution(input_dim=FLAGS.hidden,output_dim=FLAGS.hidden2,adj=self.adjs[v],act=lambda x:x,dropout=self.dropout,
                                                  logging=self.logging,name='e_dense_2_'+str(v))(self.noise)
                self.embeddings.append(embeddings)

        self.cluster_layer = ClusteringLayer(input_dim=FLAGS.hidden2,n_clusters=self.num_clusters,name='clustering')
        self.cluster_layer_q = self.cluster_layer(self.embeddings[FLAGS.input_view])

        #decoder1:将encoder中所有视图的embedding作为输入
        self.reconstructions = []
        for v in range(self.numView):
            view_reconstruction = InnerProductDecoder(input_dim=FLAGS.hidden2,name='e_weight_single_',v=v,act=lambda x:x,logging=self.logging)(self.embeddings[v])
            self.reconstructions.append(view_reconstruction)
        #decoder2:将encoder中 max modularity的视图作为embedding
        self.reconstructions_fuze = []
        for v in range(self.numView):
            view_reconstruction = InnerProductDecoder(input_dim=FLAGS.hidden2,name='e_weight_multi_',v=v,act=lambda x:x,logging=self.logging)(self.embeddings[FLAGS.input_view])
            self.reconstructions_fuze.append(view_reconstruction)


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

class ClusteringLayer(Layer):
    def __init__(self,input_dim,n_clusters=3,weight=None,alpha=1.0,**kwargs):
        super(ClusteringLayer,self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weight
        self.vars['clusters'] = weight_variable_glorot(self.n_clusters, input_dim,name="cluster_weight")

    def _call(self, inputs):
        #计算 student's t分布
        q = tf.constant(1.0)/ (tf.constant(1.0) + tf.reduce_sum(tf.square(tf.expand_dims(inputs,axis=1)-self.vars['cluster']),axis=2))
        with tf.Session() as sess:
            print("student's t distribution :qij",sess.run(q))
        q = tf.pow(q,tf.constant((self.alpha+1)/2.0))
        q = tf.transpose(tf.transpose(q)/tf.reduce_sum(q,axis=1))
        return  q

class InnerProductDecoder(Layer):
    def __init__(self, input_dim, name, v=0, dropout=0,act=tf.nn.sigmoid,**kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        with tf.variable_scope('view',reuse=tf.AUTO_REUSE):
            self.vars['view_weights'] = tf.get_variable(name=name+str(v),shape=[input_dim,input_dim],trainable=True)
            print('self.vars:view_weight',self.vars['view_weights'])
        self.dropout = dropout
        self.act =act
    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs,1-self.dropout)
        x = tf.transpose(inputs)
        tmp = tf.matmul(inputs,self.vars['view_weights'])
        x = tf.matmul(tmp, x)
        outputs = self.act(x)
        return outputs