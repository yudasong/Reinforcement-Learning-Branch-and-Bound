from pyibex import *
from BB import *
import numpy as np

#Define a Function
f = Function("x", "y", "x+y")
#Define the input domain of the function
input_box = IntervalVector(2, [0.5,1])
print(len(input_box))
#Define the output range (i.e. desired value of the function)
output_range = Interval(1,1)

test = BB(f, input_box, output_range)

print(test.getRoot(10))
