from MCTS import MCTS
import numpy as np
from main import args
from naive.BB import BB
from naive.pytorch.NNet import NNetWrapper as nn

from pyibex import *

THRESHOLD = 0.0001

f = Function("x", "y", "ln(0.5* x^2 + y^2)")
#Define the input domain of the function -- both[0.5,5] for x and y
input_box = IntervalVector(2,[-3,3])
#Define the output range (i.e. desired value of the function) -- f range [1,1]
output_range = Interval(-3,3)

game = BB(f, input_box, output_range)
nnet = nn(game)

nnet.load_checkpoint('temp/', 'best.pth.tar')
mcts = MCTS(game, nnet, args)
current_box = input_box
board = game.getBoardFromInput_box(current_box)

r = game.getGameEnded(current_box, THRESHOLD)
while r == 0:
    pi, reward = mcts.getActionProb(current_box)
    pi = game.getValidMoves(current_box, THRESHOLD) * pi
    a = np.argmax(pi)
    current_box = game.getNextState(current_box, a)
    print(current_box)
    r = game.getGameEnded(current_box, THRESHOLD)

currentValue = [[current_box[i].diam()/2 + current_box[i][0],current_box[i].diam()/2 + current_box[i][0]] for i in range(len(current_box))]
print('result value: ' + str(f.eval(IntervalVector(currentValue))[0]))
