from pyibex import *
from BB import *
import numpy as np

#Define a Function
f = Function("x", "y", "x*exp(sin(x-y))")
#Define the input domain of the function
input_box = IntervalVector(2, [0.5,5])
print(len(input_box))
#Define the output range (i.e. desired value of the function)
output_range = Interval(1,1)

test = BB(f, input_box, output_range)

assert((test.getRoot() == np.array([[0.5,  2.75, 5.  ],
 						  		   [0.5,  2.75, 5.  ]])).all())

assert((test.getNextState(0,3)[1] == np.array([2.75,       3.875,      5.        ])).all())