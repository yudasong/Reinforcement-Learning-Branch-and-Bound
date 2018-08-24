from Coach import Coach
from naive.BB import BB
from naive.pytorch.NNet import NNetWrapper as nn
from utils import *
from pyibex import *
import numpy as np
from scipy import optimize
#from BB import BB

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})
def func(x):
    #np.log(0.5*x[0]**2 + x[1]**2)
    return x[0]**2 + 3* x[1] **2

if __name__=="__main__":

    f = Function("x", "y", "ln(0.5* x^2 + y^2)")
    #Define the input domain of the function -- both[0.5,5] for x and y
    input_box = IntervalVector([[-3,2], [-2,3]])
    #Define the output range (i.e. desired value of the function) -- f range [1,1]
    output_range = Interval(-3,3)

    g = BB(f, input_box, output_range,func)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
