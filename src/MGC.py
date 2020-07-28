import tensorflow as tf
import scipy.io as sio
import numpy as np
from .model import ARGA
class Clustering_Runner():
    def __init__(self,data_name,n_clusters,iterations):
        print("Clustering on dataset: %s, number of iteration: %3d" % (data_name,  iterations))

        self.data_name = data_name
        self.warm_iteration = 0
        self.iterations = 50
        self.kl_iterations = 30
        self.n_clusters = n_clusters
        self.tol = 0.001
        self.time = 5

    def erun(self):
        tf.reset_default_graph()
        feas = format_data(self.data_name)
        placeholders = {
            'features':tf.placeholder(tf.float32),
            'adjs':tf.placeholder(tf.float32),
            'adjs_orig':tf.placeholder(tf.float32),
            'dropout':tf.placeholder_with_default(0.,shape=()),
            'attn_drop':tf.placeholder_with_default(0.,shape=()),
            'ffd_drop':tf.placeholder_with_default(0.,shape=()),
            'pos_weights':tf.placeholder(tf.float32),
            'fea_pos_weights':tf.placeholder(tf.float32),
            'p':tf.placeholder(tf.float32),
            'norm':tf.placeholder(tf.float32),
        }
        model = ARGA(placeholders, numView, num_features, num_clusters)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def format_data(dataset):
    datapath = ""
    data = sio.loadmat(datapath)
    truelabels, truefeatures = data['lable'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    rownetworks = np.array([data['PAP'].tolist(), data['PLP'].tolist()])
    numView = rownetworks.shape[0]
    y = truelabels
    train_idx =  data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    train_mask = sample_mask(train_idx,y.shape[0])
    val_mask = sample_mask(val_idx,y.shape[0])
    test_mask = sample_mask(test_idx,y.shape[0])
    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    adjs_orig = []
    for v in range(numView):
        adjs_orig = rownetworks[v]



if __name__ == '__main__':
    dataname = ""
    runner = Clustering_Runner()
    scores = runner.erun()
    print(scores)