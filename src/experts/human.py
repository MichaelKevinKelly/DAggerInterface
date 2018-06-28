import pyautogui
import numpy as np

##############################
## Human trackpad interface ##
##############################
class Human(object):

	def __init__(self):
		self.game_width = 800
		self.game_half_width = self.game_width / 2
		self.sensing_edge = (pyautogui.size()[0] - self.game_width) / 2
		self.max_theta = 3.14

	def get_action(self, observation):
		x = pyautogui.position()[0] - self.sensing_edge
		if x < 0: x = 0
		elif x > self.game_width: x = self.game_width
		theta = (x - self.game_half_width) / self.game_half_width * self.max_theta
		a = [theta]
		return [observation[0]*2.]