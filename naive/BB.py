import numpy as np
import pyibex as pi
from scipy import optimize
import copy

class BB():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, function, input_box, output_range, func):
        self.function = function
        self.input_box = input_box
        self.output_range = output_range
        self.contractor = pi.CtcFwdBwd(self.function, self.output_range)

        # size of representation for each variable
        self.embedding_size = 9
        self.func = func

        self.diam = input_box[0].diam() / 2


        self.lower = self.function.eval(self.input_box)[0]
        self.upper = self.function.eval(self.input_box)[1]

        #print(self.lower, self.upper)

    def getRoot(self):
        """
        Returns:
            : a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """

        return getBoardFromInput_box(self.input_box)



    def getBoardFromInput_box(self, currentInput_box):

        x,y = self.getBoardSize()
        embedding = np.zeros((x,3))

        for i in range(x):
            lower = currentInput_box[i][0]
            upper = currentInput_box[i][1]
            middle = float((lower + upper) / 2)

            embedding[i,0] = lower
            embedding[i,1] = middle
            embedding[i,2] = upper

        data_point = embedding.transpose();
        eps = np.sqrt(np.finfo(float).eps)
        derivative = []
        for x in data_point:
            a_s = []
            b_s = []
            x = optimize.approx_fprime(x, self.func, eps)
            for a in x:
                b = 0
                while np.abs(a) > 1:
                    a /= 10
                    b += 1
                a_s.append(a)
                if b == 0:
                    b_s.append(b)
                else:
                    b_s.append(np.log(b))
            derivative.append(a_s)
            derivative.append(b_s)
        result = np.concatenate((embedding, np.asarray(derivative).transpose()),axis = 1)

        x,y = self.getBoardSize()
        for i in range(x):

            result[i,0] /= self.diam
            result[i,1] /= self.diam
            result[i,2] /= self.diam

        return result


    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return  len(self.input_box),self.embedding_size

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 2*len(self.input_box)

    def getNextState(self, currentInput_box, action):
        """
        Input:
            state: current state
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        var_index = int(action / 2)
        direction = action % 2

        # split the interval of the selected variable ([(),()],[(),()])
        #print(currentInput_box)
        new_boxes = currentInput_box.bisect(var_index, 0.5)

        # choose go to half of the interval
        currentInput_box = new_boxes[direction]

        #self.contractor.contract(currentInput_box)

        #TODO: return the STATE

        return currentInput_box



    def getValidMoves(self,currentInput_box, threshold):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """

        mask = np.zeros(self.getActionSize())

        for i in range(len(currentInput_box)):
            if currentInput_box[i].diam() > threshold:
                mask[2*i] = 1
                mask[2*i+1] = 1

        return mask

    def distortInputbox(self, currentInputbox):
        newBox = []
        for i in range(len(currentInputbox)):
            cur = []
            delta = currentInputbox[i].diam()
            cur.append(currentInputbox[i][0] + (-1) ** (np.random.randint(0,2)) * delta * np.random.sample() / 10)
            cur.append(currentInputbox[i][1] + (-1) ** (np.random.randint(0,2)) * delta * np.random.sample() / 10)
            newBox.append(cur)

        return pi.IntervalVector(newBox)


    def getGameEnded(self,currentInput_box, threshold):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """


        # TODO: situation for empty box
        #if currentInput_box.is_empty():
        #    return 1000


        if 1 not in self.getValidMoves(currentInput_box, threshold):
            currentValue = [[currentInput_box[i].diam()/2 + currentInput_box[i][0],currentInput_box[i].diam()/2 + currentInput_box[i][0]] for i in range(len(currentInput_box))]
            #print(pi.IntervalVector(currentValue)[0])
            #print(self.function.eval(pi.IntervalVector(currentValue))[0])
            r = self.function.eval(pi.IntervalVector(currentValue))[0]

            return r
        else:
            return 0

    def getGameEndedTree(self,currentInput_box, threshold):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """


        # TODO: situation for empty box
        #if currentInput_box.is_empty():
        #    return 1000


        if 1 not in self.getValidMoves(currentInput_box, threshold):
            currentValue = [[currentInput_box[i].diam()/2 + currentInput_box[i][0],currentInput_box[i].diam()/2 + currentInput_box[i][0]] for i in range(len(currentInput_box))]
            #print(pi.IntervalVector(currentValue)[0])
            #print(self.function.eval(pi.IntervalVector(currentValue))[0])
            r = self.function.eval(pi.IntervalVector(currentValue))[0]

            if r > 0:
                r = - r / self.upper
            else:
                r = r / self.lower
            return r
        else:
            return 0



    def stringRepresentation(self, currentInput_box):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        string = ''.join(str(x) for x in currentInput_box)
        return string
