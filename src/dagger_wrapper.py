from dagger import DAgger

############################
## API for Julia / PyCall ##
############################
class DAggerWrapper(object):

	def __init__(self,path):
		self.path = path
		self.dagger_instance = DAgger(path)

	def get_action(self,o):
		return self.dagger_instance.get_action(o)

	def train(self):
		self.dagger_instance.train()

	def reset(self,path):
		self.close()
		self.dagger_instance = DAgger(path)

	def close(self):
		self.dagger_instance.close()