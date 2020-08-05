import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

class OptimizerAR(object):
    def __init__(self,model,preds_fuze,preds,p,labels,numView,pos_weights,fea_pos_weight,norm):
        # pred_fuze :decoder2
        # preds :decoder1
        # labels = adjs_orig 使用原来的矩阵作为监督项
        # p  目标分布
        #pos_weights
        self.cost = 0
        self.cost_list = []
        all_variables = tf.trainable_variables()

        self.l2_loss = 0
        for v in range(numView):
            self.cost += norm[v]*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.reshape(preds_fuze[v],[-1]),targets=tf.reshape(labels[v],[-1]),pos_weight=pos_weights[v]))
            # target*-log(sigmoid(logits))*pos_weight+(1-target)*-log(1-sigmoid(logits))
            # Lr = reduce_meam(A(m),A~(m))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, betal=0.9, name='adam')

        q = model.cluster_layer_q
        kl_loss = tf.reduce_sum(p*tf.log(p/q))
        self.cost_kl =