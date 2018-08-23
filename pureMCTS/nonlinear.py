import 
import generateFunctions as generator

class Nonlinear(): 
    def __init__(self, function, interval_vector, output_range):
        self.function = function
        self.interval_vector = interval_vector
        self.output_range = output_range
