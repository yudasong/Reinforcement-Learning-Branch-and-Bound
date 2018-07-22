import numpy as np
import pyibex as pi


class BB():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, function, input_box, output_range):
        self.function = function
        self.input_box = input_box
        self.output_range = output_range
        self.contractor = pi.CtcFwdBwd(self.function, self.output_range)

        # size of representation for each variable
        self.embedding_size = 3

    def getRoot(self):
        """
        Returns:
            : a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """

        shape = self.getBoardSize()
        
        embedding = np.zeros(shape)

        for i in range(shape[0]):
        	lower = self.input_box[i][0]
        	upper = self.input_box[i][1]
        	middle = float((lower + upper) / 2)

        	embedding[i,0] = lower
        	embedding[i,1] = middle
        	embedding[i,2] = upper

        return embedding

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
        return 2**len(self.input_box)

    def getNextState(self, state, action):
        """
        Input:
            state: current state
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        var_index = int(action / 2)
        direction = action % 2

        print(self.input_box)

        new_boxes = self.input_box.bisect(var_index, 0.5)

        self.input_box = new_boxes[direction]

        self.contractor.contract(self.input_box)

        #TODO: return the STATE

        return self.getRoot()



    def getValidMoves(self, threshold):
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

        for i in range(len(self.input_box)):
            if self.input_box[i].diam() > threshold:
                mask[2*i] = 1
                mask[2*i+1] = 1

        return mask


    def getGameEnded(self, threshold):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """

        if 1 not in self.getValidMoves(threshold):
            return True
        else:
            return False
        


    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
