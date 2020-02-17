import datetime
from mesa import Agent, Model, space
from mesa import time as tm


class Station(Agent):

	def __init__(self, model, number, name, location):
		super().__init__(self, model)
		self.station_number = number
		self.type = 'station'
		self.name = name
		self.location = location
		self.timer = datetime.datetime(100, 1, 1, 0, 0, 0)

	def step(self):
		pass

	def move(self, new_position):
		self.model.grid.move_agent(self, new_position)
