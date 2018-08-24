#This file will generate functions in polynomials with a given monomial number
import numpy as np
import random
import matplotlib.pyplot as plt

class GFwithMonomial():
	#the initial function taking 4 inputs
	def __init__(self, x_vector, high_degree_vector, rangeLow, rangeHigh, monomials):
		#the input processing
		self.x_vector = x_vector
		if(len(self.x_vector) != len(high_degree_vector)):
			raise Exception('size of degree vector should be same as number of variables')
		self.high_degree_vector = high_degree_vector
		self.rangeLow = rangeLow
		self.rangeHigh = rangeHigh

		self.functionString = ""
		self.monomials = monomials

	def generate(self):
		#allowed values for the highest degree and others can be zeros
		allowed_values = list(range(self.rangeLow,self.rangeHigh))
		#print(allowed_values)
		allowed_values.remove(0)
		# generate the coefficient and degree matrix
		degree_matrix = [[0 for x in range(self.monomials)] for y in range(len(self.x_vector))]
		for i in range(len(self.x_vector)):
			highestVar = self.high_degree_vector[i]
			degree_matrix[i] = np.random.randint(low=0,high=highestVar+1,size=self.monomials) # coefficient of the x for each degree
			degree_matrix[i][np.random.randint(low=0,high=self.monomials)] = highestVar #make sure one is the highest
		coe = np.random.choice(allowed_values,size=self.monomials)
		print(degree_matrix)

		#generate the stringRepresentation
		for i in range(self.monomials):
			add = ""
			if(coe[i] > 0):
				add = "+" + add
			add += str(coe[i]);
			for j in range(len(self.x_vector)):
				if(degree_matrix[j][i] == 0):
					continue
				add += "*" + self.x_vector[j]
				if(degree_matrix[j][i] > 1):
					add += "^" + str(degree_matrix[j][i])
			self.functionString += add
		return self.functionString
p = GFwithMonomial(["x1","x2","x3"],[3,4,5],-5,5, 4)
function = p.generate()
print(function)
