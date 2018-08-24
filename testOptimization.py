# this file is used to test the global optimization using scipy three methods
from scipy.optimize import basinhopping
from scipy.optimize import brute
from scipy.optimize import rosen, differential_evolution
from GFwithMonomial import GFwithMonomial
import numpy as np
import random

# TEST generated 3d function f = 2*x1^3*x2^3*x3^3-5*x1*x3-5*x2^4*x3-2*x1^3*x3^5
class optimization():

	def func(self, x):
		#f = np.power(x[0],2) + np.power(x[1],2) + np.power(x[2],2)
		f = 2 * np.power(x[0],3) * np.power(x[1],3) * np.power(x[2],3) - 5 * x[0] * x[2] - 5 * np.power(x[1],4) * x[2] - 2 * np.power(x[0],3) * np.power(x[2],5)
		return f

	def basinhopping_opt(self):
		#minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
		x0 = [-1000000000,-1000000000,-1000000000]
		result = basinhopping(self.func, x0)
		print("global minimum: x = [%.4f, %.4f, %.4f], f(ini_guess) = %.4f" % (result.x[0],result.x[1],result.x[2], result.fun))
		#global minimum: x = [900855863.9998, -1899308697.5270, 3069649694.9555]
		#f(ini_guess) = -289767082547134003478347253256009015498519474034963560230047619738747771411939459072.0000

	def brute(self):
		rranges = (slice(-100,100,10), slice(-100,100,10), slice(-100,100,10))
		resbrute = brute(self.func, rranges, full_output=True)
		print(resbrute[0])
		print(resbrute[1])


	def differential(self):
		bounds = [(-1000000000, 1000000000), (-1000000000,1000000000), (-1000000000,1000000000)]
		result = differential_evolution(self.func, bounds)
		print(result.x)
		print(result.fun)



opt = optimization()
opt.basinhopping_opt()
opt.brute()
opt.differential()
