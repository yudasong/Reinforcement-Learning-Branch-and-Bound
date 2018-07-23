import numpy as np
from pyibex import *
class BB():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, function, input_box, output_range, stateSize):
        self.function = function
        self.input_box = input_box
        self.output_range = output_range
        self.contractor = CtcFwdBwd(self.function, self.output_range)
        self.stateSize = stateSize

    def getRoot(self):

        """
        Returns:
            : a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """

        return getBoardFromInput_box(self.input_box)

    def getBoardFromInput_box(self,currentInput_box):
        res = np.zeros((self.stateSize, self.stateSize))
        horizentalDiff = currentInput_box[0].diam()/(self.stateSize-1)
        verticalDiff = currentInput_box[1].diam()/(self.stateSize-1)
        horizentalStart = currentInput_box[0][0]
        verticalStart = currentInput_box[1][0]
        horizentalEnd = currentInput_box[0][1]
        verticalEnd = currentInput_box[1][1]
        for i in range(0, self.stateSize):
            for j in range(0,self.stateSize):
                currentCoordinate = IntervalVector([[horizentalStart+i * horizentalDiff ,horizentalStart+i * horizentalDiff  ],  [verticalStart+j * verticalDiff ,verticalStart+j * verticalDiff ]] )
                res[i][j] = self.function.eval(currentCoordinate)[0]
        return res

    def getBoardSize(self):

        """Returns:
            (x,y): a tuple of board dimensions"""

        return (self.stateSize, self.stateSize)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 2**len(self.input_box)

    def getNextState(self, currentInput_box, action):
        """
        Input:
            state: current state
            action: action taken by current player

        Returns:
            currentInput_box: the input_box after contract
        """
        var_index = int(action / 2)
        direction = action % 2
        new_boxes = currentInput_box.bisect(var_index, 0.5)

        currentInput_box = new_boxes[direction]

        self.contractor.contract(currentInput_box)

        return currentInput_box




    def getValidMoves(self, currentInput_box, threshold):
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


    def getGameEnded(self,currentInput_box, threshold):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. positive if value close to 0 otherwise negative

        """

        if 1 not in self.getValidMoves(currentInput_box, threshold):
            currentValue = [[currentInput_box[i].diam()/2 + currentInput_box[i][0],currentInput_box[i].diam()/2 + currentInput_box[i][0]] for i in range(len(currentInput_box))]

            return 1- np.abs(self.function.eval(IntervalVector(currentValue))[0])
        else:
            return False



    def stringRepresentation(self,currentInput_box):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        string = ''.join(str(x) for x in currentInput_box)
        return string
