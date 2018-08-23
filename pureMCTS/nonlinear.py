# the nonlinear class to solve problems 
import pyibex as pi
import generateFunctions as generator
import numpy as np 

class Nonlinear(): 
    def __init__(self, function, input_range, output_range):
        self.function = function
        self.input_range = input_range
        self.output_range = output_range

    #return a matrix 
    def getRootMatrix(self,input_range):
        num_row = len(input_range)*2 
        num_col = 2 
        matrix = np.zeros((num_row,num_col))

        for i in range(num_row):
            lower = input_range[i][0]
            upper = input_range[i][1]

            matrix[i,0] = lower
            matrix[i,1] = upper 
        return matrix
    
    



