#This file will generate functions in polynomials 
import numpy as np
import random 
import matplotlib.pyplot as plt 

class generateFunctions():
	#the initial function taking 4 inputs 
	def __init__(self, x_vector, high_degree_vector, rangeLow, rangeHigh): 
		#the input processing
		self.x_vector = x_vector 
		self.high_degree_vector = high_degree_vector
		self.rangeLow = rangeLow
		self.rangeHigh = rangeHigh

		self.function = 0 

	def generate(self):
		#allowed values for the highest degree and others can be zeros 
		allowed_values = list(range(self.rangeLow,self.rangeHigh))
		allowed_values.remove(0)
		for i in range(len(self.x_vector)):
			highestVar = self.high_degree_vector[i]
			ppar = np.random.randint(low=self.rangeLow,high=self.rangeHigh,size=(highestVar+1))
			#make sure the highest is not zero coefficient
			if ppar[0] == 0: 
				ppar[0] = random.choice(allowed_values)
			for j in range(len(ppar)): 
				self.function += ppar[j]*(self.x_vector[i]**(highestVar-j))
		return self.function 

p = generateFunctions([x1,x2],[3,3],-5,5)
function = p.generate()
print(function)



