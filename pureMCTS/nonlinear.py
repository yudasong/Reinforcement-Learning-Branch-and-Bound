# the nonlinear class to solve problems 
import pyibex as pi
import generateFunctions as generator
import numpy as np 
import pureMCTS

BISECT = 0.5
THRESHOLD  = 0.001
class Nonliner: 
    def __init__(self, function, input_range, output_range):
        self.function = function
        self.input_range = input_range
        self.output_range = output_range
        self.contractor = CtcFwdBwd(self.function, output_range)
        self.num_var = len(input_range)

    #return a matrix that begins with 
    def getRootMatrix(self):
        self.input_range = self.getValidRange(self.input_range)
        #the case where does not exsit a solution at first 
        haveSolution = not self.emptyRange(self.input_range)
        if haveSolution: 
            matrix = self.convertToMatrix(self.input_range)
            return matrix
        else: 
            return None
    
    #contract
    def getValidRange(self,range):
        self.contractor.contract(range)
        return range
    
    def emptyRange(self,range):
        if len(range) == 0: 
            return True
        else: 
            return False
    
    #return a list of pairs of actions and matrix
    def getValidMoves(self, vector):
        pairs_list = []
        if not vector.is_empty():
            for i in range(len(vector)):
                if vector[i].diam() > THRESHOLD:
                    new_ranges = vector.bisect(i,BISECT)
                    action_num = i+1 
                    action_pair_lower = (action_num,0)
                    action_pair_upper = (action_num,1)
                    pairs_list.append((new_ranges[0],action_pair_lower))
                    pairs_list.append((new_ranges[1],action_pair_upper))
        return pairs_list

    #return an intervalVectors back 
    def convertToVector(self, matrix):
        vector = IntervalVector(num_var)
        for i in range(num_var):
            lower = matrix[i][0]
            upper = matrix[i][1]
            vector[i] = [lower,upper]
        return vector
    
    #return a matrix from the vectors 
    def convertToMatrix(self,vector):
        num_row = self.num_var
        num_col = 2 
        matrix = np.zeros((num_row,num_col))

        for i in range(num_row):
                lower_col = self.input_range[i][0]
                upper_col = self.nput_range[i][1]

                matrix[i,0] = lower_col
                matrix[i,1] = upper_col 
        return matrix 
    
    #true means that it is not finished 
    #false means finished 
    def notFinished(self,vector):
        for i in range(len(vector)):
            if vector[i].diam() > THRESHOLD:
                return True
        return False
    

tryObject = Nonlinear(function,input_range,output_range)
rootMatrix = tryObject.getRootMatrix()
if rootMatrix ==  None: 
    print("There is no solution")
else: 
    mcts = pureMCTS(tryObject)
    
    




