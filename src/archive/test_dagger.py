from dagger import DAgger
import numpy as np
import time

class Tester(object):

	def __init__(self,algo):
		self.algo = algo
		return

	def run(self):
		
		num_episodes = 2
		num_steps = 100

		for i in range(num_episodes):
			for j in range(num_steps):
				# print('Getting action....')
		# 		# time.sleep(1)
				obs = np.random.rand(2) 
				print('Observation: {}'.format(obs))
				print('DAgger action: {}'.format(self.algo.get_action(obs)))
				# print('Done getting action....')
		# 		# time.sleep(1)
			self.algo.train()
		print(self.algo.novice.dataset_X)
		print(self.algo.novice.dataset_Y)
		return 15

def t():
	return 10

def main():
	algo = DAgger(None)
	t = Tester(algo)
	return t.run()
	
if __name__=='__main__':
	main()



