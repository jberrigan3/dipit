import random
import sys
from decision import *
from mesa import Agent, Model, space
from mesa import time as tm

MAX_DWELL_TIME = 60*60
MIN_COOK_TIME = 15*60
REQUIRED_DIP_TIME = 240*60
PASS_RATE = 0.95
MOVEMENT_TIME = 10
BUFFER = 4*60
STATION_TIMES = {'penetrant': 50, 'emulsify': 55, 'developer': 40, 'post_rinse': 55, 'pre_rinse': 61}

class Robot(Agent):
	"""
	Robot class represents decision maker. Has attributes for tracking its current
	location and current blade it is holding. Has methods for making reading/interpreting
	the state space and for making decision.
	It reads data from the manufmodel object to interpret the state and passes its action
	data back to manufmodel.
	When a decision is made, the reward is also calculated in this class
	"""
	def __init__(self, model):
		super().__init__(self, model)
		self.available = True
		self.show_deciding = True
		self.current_station = -1
		self.current_blade = -1
		self.type = 'robot'
		self.rinse_seq = False
		p1 = open('station_capacities.txt', 'r').read()
		self.capacities = eval(p1)
		p2 = open('action_times.txt', 'r').read()
		STATION_TIMES = eval(p2)


	def advance_clock(self, action):
		if action == 1:
			self.model.schedule.time += (STATION_TIMES['penetrant'] + (MOVEMENT_TIME))
		elif action == 4:
			self.model.schedule.time += (STATION_TIMES['emulsify'] + STATION_TIMES['post_rinse'] + MOVEMENT_TIME)
			self.model.schedule.time += (STATION_TIMES['emulsify'])
		elif action == 6:
			self.model.schedule.time += (STATION_TIMES['emulsify'] + STATION_TIMES['post_rinse'] + MOVEMENT_TIME)
		elif action == 7:
			self.model.schedule.time += (STATION_TIMES['developer'] + MOVEMENT_TIME)
		elif action == 3:
			self.model.schedule.time += (STATION_TIMES['pre_rinse'] + MOVEMENT_TIME)
		elif action in [5, 8, 9]:
			self.model.schedule.time += (STATION_TIMES['post_rinse'] + MOVEMENT_TIME)
		elif action != 10:
			self.model.schedule.time += MOVEMENT_TIME
		else:
			self.model.schedule.time += 5


	def get_reward(self, state, action):
		"""
		Inputs a state and and action and outputs a corresponding reward.
		Many of the states are specific and there may not be a way around
		inputting these one by one. A possibility for exploration is a reward
		generating function
		"""
		reward = -10

		if state[1] == 1: # penetrant dip occupied
			if action != 2: # action other than remove from PD
				reward = -100
			elif action == 2: # action is remove from PD
				reward = 10

		elif state[-5] == 1 and (state[4] + sum(state[6:8])) == 0 and state[1] == 0: # redip necessary and no blades advancing
			reward = -100 if action != 1 else 100 # reward if action=dip, else penalty
		elif state[-3] == 1: # if oven needs service
			if action == 9 and (state[4] + sum(state[6:8])) == 0 and state[8] > 0 and not state[-1]: # if unload is needed
				reward = 100
			if action == 11 and state[-1]:
				reward = 10

			elif (state[4] + sum(state[6:8])) == 1: # if there is a blade currently advancing
				# reward for correct action
				if (state[4] and action == 5) or (state[6] and action == 7) or (state[7] and action == 8):
					reward = 10
		else:
			if action == 1: # d2->pd
				# first dip, dwell occupied, PD empty, and no blade currently advancing
				if state[-2] == 1 and state[2] > 0 and state[1] == 0 and (state[4] + sum(state[6:8])) == 0:
					reward = 10
			elif action == 0: # q->d2
				# queue occupied, PD empty, space in dwell
				if state[0] > 0 and state[1] == 0 and state[2] < 2 and (state[4] + sum(state[6:8])) == 0:
					reward = 10
			elif action == 2: # pd->d2
				# PD occupied
				if state[1] > 0:
					reward = 10
			elif action == 3: # d2->pre
				# dwell occupied, none currently advancing, advancing possible, oven space available
				if (state[2] > 0) and (sum(state[3:5]) + sum(state[6:8])) == 0 and state[-4] and not state[-1]:
					reward = 100
			elif action == 4: # pre->ed
				# pre rinse occupied, blade advancing, oven space available
				if (state[3] > 0) and sum(state[4:8]) < 1 and not state[-1]:
					reward = 100
			elif action == 5: # ed->post
				# ED occupied, blade advancing, oven space available
				if (state[4] > 0) and sum(state[5:8]) == 0 and not state[-1]:
					reward = 100
			elif action == 6: # post->I
				# PostR occupied, blade advancing, oven space available
				if (state[5] > 0) and (state[4] + sum(state[6:8])) == 0 and state[8] < 2 and not state[-1]:
					reward = 100
			elif action == 7: # I->dd
				# Inspection occupied, blade advancing, oven space available
				if (state[6] > 0) and (state[4] + sum(state[6:8])) == 1 and state[8] < 2 and not state[-1]:
					reward = 100
			elif action == 8: # dd->O
				# DD occupied, blade advancing, oven space available
				if (state[7] > 0) and (state[4] + sum(state[6:8])) == 1 and state[8] < 2 and not state[-1]:
					reward = 100
			elif action == 9: # O->unload
				reward = -100
			else: # idle
				# oven or dwell occupied, advancing not occurring or possible, no dips needed
				if (state[2] == 2 or (state[8] > 0 and not state[0] and not state[5] and not state[3]) or (state[2] > 0 and state[0] == 0)) and not state[-4] and (state[4] + sum(state[6:8])) == 0 and not (state[-2] or state[-5] or state[-1]):
					reward = 10
					if (state[8] > 0 and (state[5] or state[3])):
						reward = -10
				# oven full but removal not needed, redip not needed, dwell occupied
				elif state[-3] == 0 and state[-5] == 0 and state[8] == 2 and state[2] > 0 and (state[4] + sum(state[6:8])) == 0 and not state[-1]:
					reward = 10
				elif state[-1] and action == 11:
					reward = 10
		if reward < 0 and not self.model.test:
			self.model.schedule.time -= 1
		return reward


	def get_state_rep(self):
		"""
		This function reads the data from the environment (manufmodel object)
		It uses this information to construct the state space representation.
		A large portion of this subroutine is code specific to the animation/
		simulation. When implementing on an actual robot the inputs will be
		sensor readings.
		"""
		# get values for all blade times in dwell/oven/etc.
		dwell = []
		oven = []
		total = []
		system = []
		for b in self.model.blades:
			if b.get_location() == 2 and b.initial_dip_time > -1:
				dwell.append(b)
			elif b.get_location() == 8:
				oven.append(b)
			if b.initial_dip_time > 0 and b.get_location() < 3:
				total.append(b)
			if b.get_location() > -1:
				system.append(b)

		dwell_time = max([self.model.schedule.time - d.get_dwell_time() for d in dwell]) + BUFFER if len(dwell) > 0 else 0
		print([self.model.schedule.time - d.get_dwell_time() for d in dwell] if len(dwell) > 0 else str(0))
		# print('dwell times: ', dwell_time)
		oven_time = max([self.model.schedule.time - o.get_oven_time() for o in oven]) if len(oven) > 0 else 0
		undipped = -1 if len(system) == 0 else min([s.initial_dip_time for s in system])
		total_time = max([self.model.schedule.time - t.initial_dip_time for t in total]) if len(total) > 0 else 0
		# print([s.initial_dip_time for s in system])

		state_representation = []
		blade_count = [b.data[-1] == -1 for b in self.model.blades]
		state_representation.append(1 if (len(blade_count) > 0 and sum(blade_count) > 0) else 0)

		# add blade counts for each station to state space array
		for i in range(1,10):
			blade_count = [b.data[-1] == i for b in self.model.blades]
			if i == 9:
				state_representation.append(1 if (len(blade_count) > 0 and sum(blade_count) == len(blade_count)) else 0)
			elif i in [2, 3, 5, 8]: # TODO: possibly change this
				if sum(blade_count) == list(self.capacities.values())[i - 1]:
					state_representation.append(2)
				elif sum(blade_count) > 0:
					state_representation.append(1)
				else:
					state_representation.append(0)
			else:
				state_representation.append(sum(blade_count) if len(blade_count) > 0 else 0)
		# add indicator variables to state space array
		state_representation.append(1) if (dwell_time > MAX_DWELL_TIME - BUFFER and state_representation[2] > 0) else state_representation.append(0)
		state_representation.append(1) if (total_time > REQUIRED_DIP_TIME and (state_representation[2] + state_representation[1]) > 0) else state_representation.append(0)
		state_representation.append(1) if (oven_time > MIN_COOK_TIME and state_representation[8] > 0) else state_representation.append(0)
		state_representation.append(1) if (undipped == -1 and state_representation[2] > 0) else state_representation.append(0)
		state_representation.append(1) if self.rinse_seq else state_representation.append(0)

		return state_representation

	def get_priority_blade(self, start, finish, s_r):
		"""
		This subroutine determines which blade to pick if an action is chosen
		for a station that is housing multiple blades. This will almost entirely
		be changed when moving to a physical robot; the logic here mostly deals
		with elements internal to the simulation
		"""
		if start == 0:
			start -= 1
		earliest = 1000000000
		priority = 0
		if finish == 3:
			for b in self.model.blades:
				if b.get_location() == start and b.initial_dip_time <= earliest and b.initial_dip_time > -1:
					earliest = b.initial_dip_time
					priority = b.number
		else:
			for b in self.model.blades:
				if b.get_location() == start and b.data[start] <= earliest:
					if start == 2:
						# if both first dip and redip needed prioritize redip
						if s_r[-1] and s_r[-4] and b.data[start] > 0:
							earliest = b.data[start]
							priority = b.number
						# else do whichever is needed
						if not (s_r[-1] and s_r[-4]):
							earliest = b.data[start]
							priority = b.number
					else:
						earliest = b.data[start]
						priority = b.number
		return priority

	def decision(self):
		"""
		This function makes a decision based on the state space, makes a call
		to determine the reward, and then updates the state values. All of the
		functions central to the algorithm are located in the decision.py file.
		"""
		# construct state space
		state_representation = self.get_state_rep()
		# Determine the action to take
		action, next_station, movements = decision(state_representation, self.model.test)
		if self.model.debug or self.model.test:
			print('---SYSTEM INFO---')
			print('Blade needs redip: ' + ('Yes' if state_representation[-5] else 'No'))
			print('Blade can be advanced: ' + ('Yes' if state_representation[-4] else 'No'))
			print('Blade needs removal from oven: ' + ('Yes' if state_representation[-3] else 'No'))
			print('Blade needs initial dip: ' + ('Yes' if state_representation[-2] else 'No'))
			print('Gripper needs rinse: ' + ('Yes' if state_representation[-1] else 'No'))
			print('---RL DATA---')
			print('Current State: ' + str(state_representation))
			print("Chosen action: " + str(action))

		# if the terminal state is reached, the action will be None
		if not action is None:
			# Determine which blade is chosen and determine the reward for the action
			blade_to_service = self.get_priority_blade(movements[0], next_station, state_representation)
			reward = self.get_reward(state_representation, action)
			if action == 2 and reward > 0:
				self.rinse_seq = True
			if action == 10 and reward > 0:
				self.model.schedule.time += 59 # TODO: move these lines into the advance clock function?
			elif action == 11 and reward > 0:
				self.rinse_seq = False
				self.model.schedule.time += 59 # TODO: why is this not working for certain numbers?
			if self.model.debug or self.model.test:
				print('Reward received: ' + str(reward))


			# update the values based on the action and reward
			valid_action = update_value(state_representation, action, reward, self.model.test)
			# if the resulting state is in the state space update appropriate simulation data
			if not valid_action:
				next_station = -1
			# this will be changed for physical robot
			if not next_station == -1:
				if action == 1:
					if self.model.blades[blade_to_service].initial_dip_time == -1:
						self.model.blades[blade_to_service].initial_dip_time = self.model.schedule.time
					self.model.blades[blade_to_service].set_dip_time(self.model.schedule.time)
				if action == 2:
					self.model.blades[blade_to_service].set_dwell_time(self.model.schedule.time)
				if action == 8:
					self.model.blades[blade_to_service].set_oven_time(self.model.schedule.time)
				self.advance_clock(action)
		else:
			self.model.running = False
			return -1, -1
		return blade_to_service, next_station


	def step(self):
		"""
		This function is called at each time step for the robot.
		The robot makes a decision at each step. Afterwards, the clock and
		simulation visualization are updated.
		"""

		blade_to_move, next_station = self.decision()
		if not next_station == -1 and not next_station is None:
			self.available = True
			self.show_deciding = False
			self.current_station = next_station
			self.current_blade = blade_to_move
			self.model.blades[self.current_blade].data[-1] = next_station
			self.model.update_grid(self.model.blades[self.current_blade])
		self.model.update_clock()
