from __future__ import print_function
from pyibex import *

#branch-and-prune loop
def solve(f, input_box, output_range):
    box_stack = [input_box]
    contractor = CtcFwdBwd(f, output_range)
    while box_stack:
        b = box_stack.pop()
        #print(b)
        contractor.contract(b)
        if not b.is_empty():
            branched = False
            for i in range(len(b)):
                if b[i].diam() > 0.001:
                    new_boxes = b.bisect(i, 0.4)
                    box_stack.append(new_boxes[0])
                    box_stack.append(new_boxes[1])
                    branched = True
                    break
            if not branched:
                return b
    return None

#Define a Function
f = Function("x", "y", "x*exp(sin(x-y))")
#Define the input domain of the function
input_box = IntervalVector(2, [0.5,5])
print(len(input_box))
#Define the output range (i.e. desired value of the function)
output_range = Interval(1,1)

#if solvable then a solution is printed; otherwise it outputs None
print("Solution Box:", solve(f, input_box, output_range))

#Plot the function here
#...
