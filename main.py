import argparse
from tqdm import tqdm
import os
import sys
from datetime import datetime
from decision import *
from manufmodel import ManufModel
from mesa import Agent, Model, space
from mesa import time as tm
from mesa.visualization.modules import ChartModule


def get_path(f_path):
	if not os.path.isabs(f_path):
		f_path = os.getcwd() + '\\' + f_path
	return f_path

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='sample command:  python main.py -b 8 \
	 								-rand -train -e 5', add_help=True)
	parser.add_argument("-b", "--blades", dest='num_blades',
						help="Number of blades to process", type=int)
	parser.add_argument("-rand", "--random-arrivals",
						help="turns on random arrivals",
						dest='random_arrivals', action='store_true')
	parser.add_argument("-batch", "--batch-arrivals",
						help="turns on batch arrivals",
						dest='random_arrivals', action='store_false')
	parser.add_argument("-rate", "--arrival-rate",
						dest="rate", help="parameter lambda for inter-arrival \
						rate between blades. Default is 0.066 for average 15 \
						minutes between arrivals.",
						default=0.06666666, type=float)
	parser.add_argument("-train", "--training-mode",
						dest="train", help="specify whether you are training or \
						testing the algorithm",
						action='store_true')
	parser.add_argument("-test", "--testing-mode",
						dest="train", help="specify whether you are training or \
						testing the algorithm",
						action='store_false')
	parser.add_argument("-e", "--episodes",
						dest="episodes", help="specify how many episodes to run",
						default=1, type=int)
	parser.add_argument("-pol", "--policy",
						dest="policy", help="provide path for policy to use if \
						running in testing mode",
						default=None)
	parser.add_argument("-val", "--values-matrix",
						dest="values", help="if continuing training with more episodes \
						and you do not want to restart, provide a saved values matrix.",
						default=None)
	parser.add_argument("-d", "--debug",
						dest="debug", help="run step by step and print values to console",
						action='store_true')
	parser.add_argument('-unv', '--estimate-unvisited', dest='estimate', help='specify if you want to estimate \
						 unvisited states after training', action='store_true')

	args = parser.parse_args()

	if args.values:
		try:
			# loading a numpy array with a previously trained Q_sa
			print('Loading values matrix...\n')
			load_values(args.values)
			random.seed(datetime.now().second)
			print('Values matrix successfully loaded\n')
		except Exception as e:
			print('Something went wrong loading the values. Exiting program\n')
			exit()

	# number of iterations for episodes to run
	print('\nTraining for ' + str(args.episodes) + ' episode(s)...\n')
	for i in range(0, args.episodes):
		model = ManufModel(args.num_blades, args.random_arrivals, args.policy, args.train, args.rate, args.debug)
		while model.running:
			x = input(' ') if args.debug else 0
			model.step()
			if model.debug or model.test:
				print('Episode: ' + str(i) + ' / ' + 'Time: ' + str(model.schedule.time // 60) + 'm ' + str(model.schedule.time % 60) + 's' + '\n')
			else:
				sys.stdout.write('Episode: %i / minute: %i\r' % (i, model.schedule.time / 60))
				sys.stdout.flush()

		save_policy(i)
		average_rewards()

	# write Q_sa and policy to files
	if args.estimate:
		estimate_unvisited()
		save_policy('est')
	save_rewards()
	save_values()
	print('\nTraining Complete. Policies saved at each iteration as policy_i.out. Values matrix stored as values.npy.')
