import numpy as np

####################################################################
## Vanilla decision rule from https://arxiv.org/pdf/1011.0686.pdf ##
####################################################################
class vanilla_dagger_decision_rule(object):

	def __init__(self, novice, expert, beta_decay, beta0 = 1.0):
		self.novice = novice
		self.expert = expert
		self.beta0 = beta0
		self.beta_decay = beta_decay
		self.epoch = 0

	def get_actions(self, observation):
		novice_action = self.novice.get_action(observation)
		expert_action = self.expert.get_action(observation)

		beta = self.beta0 * (self.beta_decay ** self.epoch)

		## Returns a tuple: (action_to_take, expert_action)
		if np.random.rand() > beta:
			return (novice_action,expert_action)
		else:
			return (expert_action,expert_action)

	def set_epoch(self, epoch):
		self.epoch = epoch
		return