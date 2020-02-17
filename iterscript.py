import os

for i in range(17, 33):
    os.system('python main.py -b ' + str(i) + ' -rand -train -rate 0.025')
