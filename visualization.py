import random
import argparse
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, BarChartModule
from manufmodel import ManufModel
random.seed(1)

def agent_portrayal(agent):
	stations = ['Dwell #1', 'Pen Dip', 'Dwell #2', 'Pre Rinse', 'Emul Dip',
				'Post Rinse', 'Inspection', 'Dev Dip', 'Oven', 'Clock']
	portrayal = {"Shape": "circle",
				 "Color": "Grey",
				 "Filled": "true",
				 "Layer": 1,
				 "r": 0.2,
				 "text": "Idle",
				 "text_color": "black"}
	portrayal['font'] = '10px Arial'

	if (agent.type == 'station'):
		if agent.name == 'Clock':
			portrayal["Color"] = 'white'
			portrayal['text'] = str(agent.timer.time())
			portrayal['text_color'] = 'black'
			portrayal['Layer'] = 0
			portrayal['r'] = 0.00000001
			portrayal['font'] = '20px Arial'
		else:
			portrayal["Color"] = 'white'
			portrayal['text'] = str(agent.name)
			portrayal['text_color'] = 'black'
			portrayal['Layer'] = 0
			portrayal['r'] = 0.00000001

	if (agent.type == 'blade'):
		portrayal["Color"] = 'blue'
		portrayal['text'] = str(agent.number)
		portrayal['text_color'] = 'white'

		portrayal['Layer'] = 5

	if (agent.type == 'robot'):
		portrayal['r'] = 1
		if (not agent.available or not agent.show_deciding):
			portrayal['Color'] = '#309c4b'
			portrayal['text_color'] = 'black'
			if (agent.current_station in [1, 4, 7]):
				portrayal['text'] = 'Moving blade ' + str(agent.current_blade) + ' and holding at ' + stations[agent.current_station]
			elif (agent.current_station in [0, 2, 8]):
				portrayal['text'] = 'Moving blade ' + str(agent.current_blade) + ' to ' + stations[agent.current_station]
			elif (agent.current_station in [3, 5]):
				portrayal['text'] = 'Moving ' + str(agent.current_blade) + ' to ' + stations[agent.current_station]
			elif (agent.current_station == 6):
				portrayal['text'] = 'Holding blade ' + str(agent.current_blade) + ' for inspection'
			else:
				portrayal['text'] = 'Unloading blade ' + str(agent.current_blade)

	return portrayal

grid = CanvasGrid(agent_portrayal, 5, 4, 600, 300)

parser = argparse.ArgumentParser(description='sample command:  python main.py -inst Atlanta.tsp \
								-alg BnB -time 10 -seed 1')
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
					default=0.066, type=float)
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

args = parser.parse_args()

model_params = {"num_blades": args.num_blades,
                "rand_arr": args.random_arrivals,
                "pol": args.policy,
				"train": args.train,
				"rate": args.rate
                }

server = ModularServer(ManufModel,
                       [grid],
                       "Manufacturing Line Model",
					   model_params = model_params
                       )
server.port = 8521 # The default
server.launch()
