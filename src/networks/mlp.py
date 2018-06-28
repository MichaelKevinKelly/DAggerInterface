import numpy as np
import tensorflow as tf

###############################
## Vanilla NN Implementation ##
###############################
class MLP(object):

    ## TODO: add batchnorm

    def __init__(self,
        sess,
        obs_dim,
        action_dim,
        name,
        hidden_sizes = [64,32,32,32],
        learning_rate = 1e-3,
        batch_size = 4,
        validation_ratio = 0.0,
        max_epochs = 500,
        dropout_rate = 0.0,
        weight_reg = 1e-3,
        ):

        self.sess = sess
        self.obs_dim = np.prod(obs_dim)
        self.action_dim = np.prod(action_dim)
        self.hidden_sizes = hidden_sizes
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_ratio = validation_ratio
        self.max_epochs = max_epochs
        self.dropout_rate = dropout_rate
        self.weight_reg = weight_reg
        self.epochs_to_save = [i*int(max_epochs/5) for i in range(5)] + [max_epochs-1]

        self.inputs = tf.placeholder(tf.float32, [None, obs_dim], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, action_dim], name='outputs')
        self.phase = tf.placeholder(tf.bool, name='phase')

        with tf.variable_scope(name) as scope:
            
            hidden = tf.layers.dense(
                self.inputs,
                hidden_sizes[0], 
                name = 'h0',
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l1_regularizer(weight_reg),
                kernel_initializer = tf.keras.initializers.he_normal())

            hidden = tf.layers.dropout(hidden,rate=dropout_rate,training=self.phase)

            for i in range(1,len(hidden_sizes)):
                
                hidden = tf.layers.dense(
                    hidden,
                    hidden_sizes[i],
                    name = 'h%d' % i,
                    activation = tf.nn.relu,
                    kernel_regularizer = tf.contrib.layers.l1_regularizer(weight_reg),
                    kernel_initializer = tf.keras.initializers.he_normal())
                
                hidden = tf.layers.dropout(hidden,rate=dropout_rate,training=self.phase)

            self.outputs = tf.layers.dense(
                hidden,
                action_dim,
                name = 'output', 
                kernel_regularizer = tf.contrib.layers.l1_regularizer(weight_reg),
                kernel_initializer = tf.keras.initializers.he_normal())

            self.loss = tf.nn.l2_loss(self.outputs - self.labels)
            self.optims = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            self.saver = tf.train.Saver(max_to_keep=None)

        return

    def load_data(self, data):
        self.dataset_X = data[0]
        self.dataset_Y = data[1]

    def split_train_test(self):
        N = self.dataset_X.shape[0]

        if(self.validation_ratio > 0.0):

            Ntrain = int(N * (1-self.validation_ratio))
            Ntest = N - Ntrain

            shuffle_inds = np.arange(N)
            np.random.shuffle(shuffle_inds)
            
            self.train_data_X = self.dataset_X[shuffle_inds[:Ntrain], :]
            self.train_data_Y = self.dataset_Y[shuffle_inds[:Ntrain], :]
            self.test_data_X = self.dataset_X[shuffle_inds[Ntrain:], :]
            self.test_data_Y = self.dataset_Y[shuffle_inds[Ntrain:], :]

        else:
            self.train_data_X = self.dataset_X
            self.train_data_Y = self.dataset_Y
            self.test_data_X = None
            self.test_data_Y = None

    def train(self, model_name=None, verbose=False):
        # if model_name == None then it will not save

        min_test_loss = np.Inf

        for epoch in range(self.max_epochs):
            Ntrain = self.train_data_X.shape[0]
            num_batches = Ntrain // self.batch_size

            # shuffle train data
            shuffle_inds = np.arange(Ntrain)
            np.random.shuffle(shuffle_inds)
            self.train_data_X = self.train_data_X[shuffle_inds,:]
            self.train_data_Y = self.train_data_Y[shuffle_inds,:]
            for idx in range(num_batches):
                batch_ind_start = idx*self.batch_size
                batch_ind_end = (idx+1)*self.batch_size
                batch_X = self.train_data_X[batch_ind_start:batch_ind_end,:]
                batch_Y = self.train_data_Y[batch_ind_start:batch_ind_end,:]

                self.sess.run(self.optims, 
                    feed_dict = dict(zip([self.inputs, self.labels, self.phase], 
                                        [batch_X, batch_Y, True])))

            if verbose: print('Epoch %d Train Loss: %.3f' % (epoch, 
                self.eval_performance(self.train_data_X, self.train_data_Y)))

            if(self.test_data_X is not None):
                test_loss = self.eval_performance(self.test_data_X, self.test_data_Y)
                if verbose: print('Epoch %d Test Loss: %.3f' % (epoch, test_loss))

            if(model_name is not None) and epoch in self.epochs_to_save:
                save_path = self.saver.save(self.sess, model_name, global_step=epoch)
                if verbose: print('Saved model in path: %s' % save_path)

    def load_model(self, model_name):
        self.saver.restore(self.sess, model_name)

    def eval(self, x):
        return self.sess.run(self.outputs, 
            feed_dict = dict(zip([self.inputs, self.phase], 
                                [x, False])))


    def eval_performance(self, x, labels):
        N = labels.shape[0]
        return self.sess.run(self.loss, 
            feed_dict = dict(zip([self.inputs, self.labels, self.phase], 
                                [x, labels, False]))) / N

    ## Run time API
    def get_action(self,x):
        x = np.expand_dims(x,axis=0)
    
        action = self.sess.run(self.outputs, 
            feed_dict = dict(zip([self.inputs, self.phase], 
                                [x, False])))

        return action[0]


# def main():

#     ## Sanity check
#     with tf.Session() as sess:
        
#         ## Set up / initialize network
#         pol = MLP(sess, 2, 1, 'policy', 
#             max_epochs = 500, hidden_sizes = [64,64,32], validation_ratio=0.2,
#             weight_reg = 1e-3, dropout_rate=0.1)
#         tf.global_variables_initializer().run()

#         ## Train on noised sine curve
#         xx = np.random.rand(50,2)
#         yy = np.expand_dims(np.mean(xx,axis=1),axis=-1)
#         pol.load_data([xx,yy])

#         # xx = np.linspace(-1,1,50).reshape((50,1))
#         # pol.load_data([xx, np.sin(xx*np.pi) + 0.02*np.random.randn(*xx.shape)])
#         np.random.seed(1)
#         pol.split_train_test()
#         pol.train(verbose=True)

#         ## Test and render
#         xx = np.random.rand(50,2)
#         yy = np.expand_dims(np.mean(xx,axis=1),axis=-1)
#         plt.plot(np.arange(50), yy)
#         plt.hold(True)
#         plt.plot(np.arange(50),pol.eval(xx))
#         plt.show()
#         return

#         xx = np.linspace(-1.5,1.5, 100).reshape((100,1))
#         plt.plot(xx, np.sin(xx*np.pi))
#         plt.hold(True)
#         yy = pol.eval(xx)
#         plt.plot(xx,yy)
#         plt.plot(pol.train_data_X, pol.train_data_Y, 'o')
#         plt.plot(pol.test_data_X, pol.test_data_Y, 'o')

#         plt.show()


# if __name__ == '__main__':

#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     main()


