from Coach import Coach
from naive.BB import BB
from naive.pytorch.NNet import NNetWrapper as nn
from utils import *
from pyibex import *
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

#-20 * exp(-0.2*sqrt(0.5*(x^2+y^2)))-exp(0.5*(cos(2*3.1415926535*x)+cos(2*3.1415926535*y)))+2.71828+20

if __name__=="__main__":
    f = Function("x", "y", "x^2-y^2")
    #Define the input domain of the function -- both[0.5,5] for x and y
    input_box = IntervalVector(2,[-5,5])
    #Define the output range (i.e. desired value of the function) -- f range [1,1]
    output_range = Interval(0,10)

    g = BB(f, input_box, output_range)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
