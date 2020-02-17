import time
import datetime
import sys
from decision import *
from blade import Blade
from station import Station
from robot import Robot
from mesa import Agent, Model, space
from mesa import time as tm
from mesa.visualization.modules import ChartModule

"""
These are the parameters that can be changed for the system
"""
random.seed(1)

class ManufModel(Model):
	"""
	Total model contains the robot along with all blades and stations. These
	are created upon initialization. It is the base model that essentially controls
	the simulation.
	"""
	def __init__(self, num_blades, rand_arr, pol, train, rate, debug = False):
		self.robot = Robot(self)
		self.debug = debug
		self.num_blades = num_blades
		self.schedule = tm.BaseScheduler(self)
		self.grid = space.MultiGrid(5, 4, True)
		self.grid.place_agent(self.robot, (2, 1))
		self.locations = self.locations = [(1,0), (2,0), (3,0), (4,0), (4,2), (3,2), (3,3), (2,2), (1,2), (0,2)]
		self.arrival_times = self.generate_arrival_times(rand_arr, rate, train)
		self.blades = [Blade(self, i, entry_time=self.arrival_times[i]*60) for i in range(0,len(self.arrival_times))]
		self.stations = [Station(self, 0, 'Dwell #1', (1,0)),
						 Station(self, 1, 'Pen Dip', (2,0)),
						 Station(self, 2, 'Dwell #2', (3,0)),
						 Station(self, 3, 'Pre Rinse', (4,0)),
						 Station(self, 4, 'Emul Dip', (4,2)),
						 Station(self, 5, 'Post Rinse', (3,2)),
						 Station(self, 6, 'Inspection', (3,3)),
						 Station(self, 7, 'Dev Dip', (2,2)),
						 Station(self, 8, 'Oven', (1,2))]
		self.clock = Station(self, 10, 'Clock', (0,3))
		self.running = True
		self.done = 0
		self.zero = 0
		self.test = not train

		# place blades at starting point
		for b in self.blades:
			if b.entry_time == 0:
				self.grid.place_agent(b, (0,0))

		# place station numbers in corresponding spots
		for s in self.stations:
		    self.grid.place_agent(s, s.location)

		self.grid.place_agent(self.clock, (0,3))
		if pol:
			if train:
				print('You can only provide a policy if you are testing the algorithm. \
				please correct the arguments and retry.\n')
				exit()
			try:
				print('Loading the supplied policy...\n')
				load_policy(pol)
				print('Policy successfully loaded.\n')
			except Exception as e:
				print('something went wrong loading the policy. Exiting program.\n')
				exit()
		if not train and not pol:
			print('Please supply a path to a policy file to test.\n')
			exit()


	def step(self):
		"""
		Increments the time in the simulation. At each time step, the Robot
		reads state, makes decision, and the state is updated.
		"""
		# update the visualization and collect data
		# print([b.entry_time for b in self.blades])
		for b in self.blades:
			if not (self.schedule.time == 0 and self.zero == 1):
				if self.schedule.time >= b.entry_time and b.data[-1] == -2:
					b.data[-1] = -1
					b.entry_time = self.schedule.time
				# if self.schedule.time in [b.entry_time, b.entry_time - 1]:
					self.grid.place_agent(b, (0,0))

		# add a robot step to the scheduler and execute then collect data
		self.schedule.add(self.robot)
		self.schedule.step()
		self.zero += 1

		# determine how many blades are finished
		finished = [b.finished for b in self.blades]

		# if all blades are finished, terminate
		if all(finished) and self.done > 1:
			self.running = False

		if all(finished):
			self.done += 1

	#update the visualization
	def update_grid(self, blade_to_update):
		blade_to_update.move(self.locations[blade_to_update.data[-1]])

	def update_clock(self):
		self.clock.timer = datetime.datetime(100, 1, 1, 0, 0, 0) + datetime.timedelta(0, self.schedule.time)
		self.clock.move(self.clock.location)

	def generate_arrival_times(self, rand_arr, rate, train):
		"""
		Function generates the arrival time of all blades based on the specified
		inter arrival time and the total number of blades allowed
		"""
		if rand_arr and train:
			arrivals = []
			total_time = 0
			while len(arrivals) < self.num_blades:
				arrivals.append(total_time)
				total_time += int(random.expovariate(rate)) # interarrival time
			return arrivals
			# return [0, 8, 11, 31, 31, 33, 71, 76, 81, 98, 100, 117]

		elif rand_arr and not train:
			# Test set
			# return [0, 8, 23, 27, 41, 46, 77, 95, 98, 104, 105, 105]
			# return [0, 39, 104, 120, 127, 127, 137, 167, 172, 191, 238, 239]
			# return [0, 35, 57, 69, 74, 79, 120, 145, 151, 157, 157, 163]
			# return [0, 6, 11, 22, 22, 24, 31, 81, 129, 196, 215, 219]
			# return [0, 8, 19, 22, 38, 64, 64, 66, 87, 112, 123, 125]
			# return [0, 34, 50, 60, 60, 70, 79, 86, 89, 117, 119, 122]
			# return [0, 0, 3, 11, 38, 45, 47, 47, 72, 81, 94, 94]
			# return [0, 9, 63, 68, 86, 91, 99, 100, 119, 121, 194, 207]
			# return [0, 10, 12, 48, 57, 87, 90, 116, 119, 134, 143, 144]
			return [0, 8, 11, 31, 31, 33, 71, 76, 81, 98, 100, 117]
		else:
			return [0] * self.num_blades
