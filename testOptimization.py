# this file is used to test the global optimization using scipy three methods
from scipy.optimize import basinhopping
from scipy.optimize import brute
from scipy.optimize import rosen, differential_evolution
from GFwithMonomial import GFwithMonomial as generator
import numpy as np
import random

class MyTakeStep(): 
	def __init__(self, stepsize=0.5):
		self.stepsize = stepsize
	def __call__(self, x):
		s = self.stepsize
		x[0] += np.random.uniform(-2.*s, 2.*s)
		x[1:] += np.random.uniform(-s, s, x[1:].shape)
		return x
	
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

	#useful that the function has many minima separated by large barriers 
	def basinhopping_opt(self):
		#minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
		x0 = [-8,-1000000000,-1000000000]
		#func: function to be optimized 
		#x0: initial guess 
		#niter: the number of basin-hopping iterations 

		minimizer_kwargs0 = {"method": "BFGS"}
		minimizer_kwargs1 = {"method":"L-BFGS-B", "jac":True}
		function = self.func
		niter_success = 40 
		stepsize = MyTakeStep()
		result = basinhopping(function, x0,  niter=100,minimizer_kwargs=minimizer_kwargs0, stepsize=stepsize.stepsize)
		print("1st: global minimum: x = [%.4f, %.4f, %.4f], f(ini_guess) = %.4f" % (result.x[0],result.x[1],result.x[2],result.fun))
		
		x1 = [2000,-1000,67]
		result2 = basinhopping(function,x1,niter=50)
		print("2nd: global minimum: x = [%.4f, %.4f, %.4f], f(ini_guess) = %.4f" % (result2.x[0],result2.x[1],result2.x[2],result2.fun))
	
		x3 = [2000,-1000,67]
		result3 = basinhopping(function,x1,niter=100)
		print("3rd: global minimum: x = [%.4f, %.4f, %.4f], f(ini_guess) = %.4f" % (result3.x[0],result3.x[1],result3.x[2],result3.fun))
	
		#x4 = [0,0,0]
		#result4 = basinhopping(function,x1,niter=100,stepsize=10,minimizer_kwargs=minimizer_kwargs1)
		#print("3rd: global minimum: x = [%.4f, %.4f, %.4f], f(ini_guess) = %.4f" % (result4.x[0],result4.x[1],result4.x[2],result4.fun))
	
	def brute(self):
		rranges = (slice(-5,5,0.25), slice(-5,5,0.25), slice(-5,5,0.25))
		function = self.func
		resbrute = brute(function, rranges, full_output=True)
		print(resbrute[0])
		print(resbrute[1])

		print("This is the second")
		resbrute2 = brute(function, rranges, full_output=True)
		print(resbrute2[0])
		print(resbrute2[1])

		print("This is the third")
		resbrute3 = brute(function, rranges, full_output=True)
		print(resbrute3[0])
		print(resbrute3[1])

		print("====================Still Brute but different ranges======================")
		rranges2 = (slice(-5,5,0.1), slice(-5,5,0.1), slice(-5,5,0.1))
		resbrute4 = brute(function, rranges2, full_output=True)
		print(resbrute4[0])
		print(resbrute4[1])

		print("This is the #5")
		resbrute5 = brute(function, rranges2, full_output=True)
		print(resbrute5[0])
		print(resbrute5[1])

		print("This is the third")
		resbrute6 = brute(function, rranges2, full_output=True)
		print(resbrute6[0])
		print(resbrute6[1])

		print("====================A much smaller range===============================")
		rranges3 = (slice(-1,1,0.1), slice(-1,1,0.1), slice(-1,1,0.1))
		resbrute7 = brute(function, rranges3, full_output=True)
		print(resbrute7[0])
		print(resbrute7[1])

		print("This is the #8")
		resbrute8 = brute(function, rranges3, full_output=True)
		print(resbrute8[0])
		print(resbrute8[1])

		print("This is the #9")
		resbrute9 = brute(function, rranges3, full_output=True)
		print(resbrute9[0])
		print(resbrute9[1])

	def differential(self):
		bounds = [(-1000000000, 1000000000), (-1000000000,1000000000), (-1000000000,1000000000)]
		function = self.func
		print("===================Differential #0==================")
		result = differential_evolution(function, bounds)
		print(result.x)
		print(result.fun)
		print("===================Differential #00==================")
		result0 = differential_evolution(function, bounds)
		print(result0.x)
		print(result0.fun)
		print("===================Differential #000==================")
		result00 = differential_evolution(function, bounds)
		print(result00.x)
		print(result00.fun)

		bounds2 = [(-1,1),(-1,1),(-1,1)]
		result2 = differential_evolution(function, bounds2)
		print("===================Differential #2==================")
		print(result2.x)
		print(result2.fun)

		result20 = differential_evolution(function, bounds2)
		print("===================Differential #20==================")
		print(result20.x)
		print(result20.fun)

		result200 = differential_evolution(function, bounds2)
		print("===================Differential #200==================")
		print(result200.x)
		print(result200.fun)

		result2000 = differential_evolution(function, bounds2)
		print("===================Differential #200==================")
		print(result2000.x)
		print(result2000.fun)


		bounds3 = [(-5,5),(-5,5),(-5,5)]
		result3 = differential_evolution(function, bounds3)
		print("===================Differential #3==================")
		print(result3.x)
		print(result3.fun)
		result30 = differential_evolution(function, bounds3)
		print("===================Differential #30==================")
		print(result30.x)
		print(result30.fun)
		result300 = differential_evolution(function, bounds3)
		print("===================Differential #300==================")
		print(result300.x)
		print(result300.fun)







opt = optimization()
opt.basinhopping_opt()
print("========================THIS IS BRUTE============================")
#opt.brute()
print("========================THIS IS DIFFERENTIAL============================")
opt.differential()
