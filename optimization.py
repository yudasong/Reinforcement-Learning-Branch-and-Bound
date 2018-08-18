from __future__ import print_function
from pyibex import *
from scipy import optimize
import numpy as np
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
def func(x):
    return x[0]**2 + x[1] ** 3

x = [1,2]
eps = np.sqrt(np.finfo(float).eps)
print(optimize.approx_fprime(x, func, [eps, np.sqrt(200) * eps]))
#Define a Function
f = Function("x1", "x2", "42.419930460509669*x1^2-25.467284450100433*x1*x2+29.037525088273682*x2^2+0.246437703822396*x1^3+0.342787267928099*x1^2*x2+0.070061019768681*x1*x2^2+0.056167250785361*x2^3-9.747135277935248*x1^4+1.281447375757236*x1^3*x2-1.066167940090009*x1^2*x2^2-0.111337393290709*x1*x2^3-3.148132699966833*x2^4-0.058675653184320*x1^5-0.088630122702897*x1^4*x2-0.035603912757564*x1^3*x2^2-0.092730054611810*x1^2*x2^3+0.030783940378564*x1*x2^4-0.016849595361031*x2^5+1.362207232588218*x1^6+1.257918398491556*x1^5*x2+0.407802497440289*x1^4*x2^2-1.168667210949858*x1^3*x2^3+1.839303562141088*x1^2*x2^4-0.729105138802864*x1*x2^5+0.326281890950742*x2^6 - 88")
#Define the input domain of the function
input_box = IntervalVector(2, [-10,10])
input_box1 = IntervalVector([[-5,3],[-3,5]])
#Define the output range (i.e. desired value of the function)
output_range = Interval(0,0)

#if solvable then a solution is printed; otherwise it outputs None
print("Solution Box:", solve(f, input_box, output_range))

#Plot the function here
#...
