# this file is used to test the global optimization using scipy three methods
from scipy.optimize import basinhopping
from scipy.optimize import brute
from scipy.optimize import rosen, differential_evolution
from GFwithMonomial import GFwithMonomial as generator
import numpy as np
import random

# TEST generated 3d function f = 2*x1^3*x2^3*x3^3-5*x1*x3-5*x2^4*x3-2*x1^3*x3^5
class optimization():

	def func(self, x):
		#print(x)
		#f = np.power(x[0],2) + np.power(x[1],2) + np.power(x[2],2)
		#f = 2 * np.power(x[0],3) * np.power(x[1],3) * np.power(x[2],3) - 5 * x[0] * x[2] - 5 * np.power(x[1],4) * x[2] - 2 * np.power(x[0],3) * np.power(x[2],5)
		p = generator(x,[3,4,5],-5,5,4)
		p.randomPara()
		f = p.generateFunction(p.coe, p.degree_matrix)
		return f

	def basinhopping_opt(self):
		#minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
		x0 = [-8,-1000000000,-1000000000]
		#func: function to be optimized 
		#x0: initial guess 
		#niter: the number of basin-hopping iterations 
		
		result = basinhopping(self.func, x0,niter=100)
		print(self.func)
		print("global minimum: x = [%.4f, %.4f, %.4f], f(ini_guess) = %.4f" % (result.x[0],result.x[1],result.x[2],result.fun))
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
#opt.brute()
#opt.differential()
