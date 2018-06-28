import tensorflow as tf
import numpy as np
import yaml
import os
from shutil import copyfile
from experts.human import Human
from networks.mlp import MLP
from decision_rules import *

## TODO: real-time memory / runtime considerations (e.g. buff, etc.)

#####################################
## DAgger framework implementation ##
#####################################
class DAgger(object):

    def __init__(self,params_path,save_path):

        ## Parameters / Data Logging Setup
        params = yaml.load(open(params_path))['parameters']
        self.save_path = save_path
        assert(not os.path.exists(self.save_path))
        os.makedirs(self.save_path)
        os.makedirs(self.save_path+'/models')
        copyfile(params_path,save_path+'/parameters.yaml')

        ## IL Defaults
        self.mode = params['mode']
        self.downsample_length = None if params['downsample_length'] < 0 else params['downsample_length']
        
        ## Optimizer / Policy Defaults
        self.name = params['name']
        self.obs_dim = int(params['obs_dim'])
        self.action_dim = int(params['action_dim'])
        self.policy_hidden_sizes = params['policy_hidden_sizes']
        self.learning_rate = float(params['learning_rate'])
        self.regularization = float(params['regularization'])
        self.batch_size = int(params['batch_size'])
        self.dropout_prob = float(params['dropout_prob'])
        self.verbose = bool(params['verbose'])
        self.validation_ratio = float(params['validation_ratio'])
        self.max_epochs = int(params['max_epochs'])

        ## Vanilla DAgger Defaults
        self.beta0 = float(params['beta0'])
        self.beta_decay = float(params['beta_decay'])

        ## Agent Setup
        self.sess = tf.Session()
        self.expert = Human()
        self.novice = MLP(self.sess,
            self.obs_dim,
            self.action_dim,
            self.name,
            hidden_sizes = self.policy_hidden_sizes,
            learning_rate = self.learning_rate,
            batch_size = self.batch_size,
            validation_ratio = self.validation_ratio,
            max_epochs = self.max_epochs,
            dropout_rate = self.dropout_prob,
            weight_reg = self.regularization
        )
        self.decision_rule = self.get_decision_rule(self.novice,self.expert)
        self.sess.run(tf.global_variables_initializer())

        ## Data
        self.observations_buff = []
        self.actions_buff = []
        self.observations = np.zeros((0,self.obs_dim))
        self.actions = np.zeros((0,self.action_dim))

    def get_action(self,observation):
        a, a_exp = self.decision_rule.get_actions(observation)
        self.observations_buff.append(observation)
        self.actions_buff.append(a_exp)
        return a

    def train(self,model_name=self.save_path+'/models/'+self.name,verbose=False):
        self.dump_data()
        self.novice.load_data((self.observations,self.actions))
        self.novice.split_train_test()
        self.novice.train(verbose=True)

    def dump_data(self):
        new_obs = np.array(self.observations_buff)
        new_acts = np.array(self.actions_buff)
        self.observations_buff = []
        self.actions_buff = []
        self.observations = np.concatenate((self.observations,new_obs))
        self.actions = np.concatenate((self.actions,new_acts))

    def get_decision_rule(self,novice,expert):
        if self.mode == 'vanilla':
            dr = vanilla_dagger_decision_rule(novice,expert,self.beta_decay,self.beta0)
        else:
            raise NotImplementedError
        return dr

    def close(self):
        tf.reset_default_graph()
        self.sess.close()
