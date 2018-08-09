from MCTS import MCTS
import numpy as np

from naive.BB import BB
from naive.pytorch.NNet import NNetWrapper as nn

from pyibex import *

THRESHOLD = 0.001

f = Function("x", "y", "x^2-y^2")
#Define the input domain of the function -- both[0.5,5] for x and y
input_box = IntervalVector(2, [-5,5])
#Define the output range (i.e. desired value of the function) -- f range [1,1]
output_range = Interval(-3,3)

game = BB(f, input_box, output_range)
nnet = nn(game)

nnet.load_checkpoint('temp/', 'best.pth.tar')

current_box = input_box
board = game.getBoardFromInput_box(current_box)

r = game.getGameEnded(current_box, THRESHOLD)

while r == 0:
    board = game.getBoardFromInput_box(current_box)
    pi, v = nnet.predict(board)
    pi = game.getValidMoves(current_box, THRESHOLD) * pi
    print(pi)
    a = np.argmax(pi)
    current_box = game.getNextState(current_box, a)
    r = game.getGameEnded(current_box, THRESHOLD)

print('result: ' + str(r))
