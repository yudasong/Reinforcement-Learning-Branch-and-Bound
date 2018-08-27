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
		self.function = 0
		self.monomials = monomials

		#random numbers generator
		self.degree_matrix = None
		self.coe = None

	def randomPara(self):
		#the range for coefficient but not included 0
		allowed_values = list(range(self.rangeLow,self.rangeHigh))
		allowed_values.remove(0)
		self.degree_matrix = [[0 for x in range(self.monomials)] for y in range(len(self.x_vector))]
		for i in range(len(self.x_vector)):
			highestVar = self.high_degree_vector[i]
			self.degree_matrix[i] = np.random.randint(low=0,high=highestVar+1,size=self.monomials) # coefficient of the x for each degree
			self.degree_matrix[i][np.random.randint(low=0,high=self.monomials)] = highestVar #make sure one is the highest
		self.coe = np.random.choice(allowed_values,size=self.monomials)

	def generateString(self,coe,degree_matrix):
		#generate the stringRepresentation
		for i in range(self.monomials):
			add = ""
			if(coe[i] > 0):
				add = "+"
			add += str(coe[i])
			for j in range(len(self.x_vector)):
				if(degree_matrix[j][i] == 0):
					continue
				add += "*" + self.x_vector[j]
				if(degree_matrix[j][i] > 1):
					add += "^" + str(degree_matrix[j][i])
			self.functionString += add
		return self.functionString

	#return a function
	def generateFunction(self,coe,degree_matrix):
		#generate the stringRepresentation
		for i in range(self.monomials):
			add = coe[i]
			for j in range(len(self.x_vector)):
				if(degree_matrix[j][i] == 0):
					continue
				add = add * (self.x_vector[j]** degree_matrix[j][i])
			self.function += add
		return self.function

	def generateFunc(self,x):
		res = 0;
		for i in range(len(self.degree_matrix[0])):
			current = self.coe[i];
			for j in range(len(self.degree_matrix)):
				current *= x[j]**self.degree_matrix[j][i]
			res += current
		return res

p = GFwithMonomial(["x1","x2","x3"],[3,4,5],-5,5,4)
p.randomPara()
print(p.coe)
print(p.degree_matrix)
print(p.generateString(p.coe, p.degree_matrix))
print(p.generateFunc([1,0,1]))
#function = p.generateString(p.coe,p.degree_matrix)
#function2 = p.generateFunction(p.coe,p.degree_matrix)
#print(function)
#print(function2)
