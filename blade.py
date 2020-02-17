from mesa import Agent, Model, space
from mesa import time as tm

class Blade(Agent):
	"""
	Blade objects are used to track blades through the system. the data attribute
	represents the necessary information needed to track blades through the system.
	data[0] = time in dwell 1
	data[1] = total time in pd
	data[3] = time sitting in dwell 2
	data[4] = time in oven
	data[5] = current blade location
	"""
	def __init__(self, model, number, entry_time=0):
		super().__init__(self, model)
		self.data = [0, -1, 0, 0, 0, 0, 0, 0, 0, -2]
		self.entry_time = entry_time #entry time into system
		self.number = number
		self.start_time = 0
		self.end_time = 0
		self.type = 'blade'
		self.finished = False
		self.initial_dip_time = -1

	def move(self, new_position):
		self.model.grid.move_agent(self, new_position)

	def get_oven_time(self):
		return self.data[8]

	def get_dwell_time(self):
		return self.data[2]

	def get_location(self):
		return self.data[-1]

	def set_oven_time(self, t):
		self.data[8] = t

	def set_dwell_time(self, t):
		self.data[2] = t

	def set_dip_time(self, t):
		self.data[1] = t

	def step(self):
		pass
