import math
import numpy as np
import copy, random 
import nonlinear as nl

# EPS = 1e-8
# THRESHOLD = 0.001
# this class intends to solve the nonlinear via pure MCTS search 
class State:
    def __init__(self, matrix, prevState, prevAction):
        self.matrix = matrix
        self.prevState = prevState
        self.prevAction = prevAction

        self.nCount = 0 
        self.qReward = 0 
        self.children = []  

class pureMCTS:
    def __init__(self, nonlinear): 
        self.over = False
        self.rootMatrix = nonlinear.getRootMatrix()
        #if the first time, there is no solution 
        if not self.rootMatrix:
            self.over  = True 
        self.root = State(rootMatrix, None, None)
        self.function = nonlinear.function
 
    def uct_searh(self):
        # computational buget
        for i in range(0,100): 
            self.over = False 
            selected_state = self.tree_policy(self.root)
            reward_state = self.default_policy(selected_state)
            self.backup(selected_state)
        action = self.best_child(self.root)         
        #if there is equal cases, just choose randomly 

    def tree_policy(self, state): 
        matrix = state.matrix
        vector = nl.convertToVector(matrix)
        if nl.notFinished(vector):
            result_state = self.expand(state)
            if not result_state: 
                result_state = self.best_child(state)
            return result_state
        else: 
            return state 

    def expand(self, state):
        #states that have been established 
        chlidren = state.children
        list_preAction = [] 
        for i in range(len(children)):
            list_preAction.append(children[i].prevAction)

        #add a heuristics to expand the minimum range first
        best_action = None 
        best_vector = None 
        best_value = math.inf
        current_vector = nl.convertToVector(state.matrix)
        list_validVectors = nl.getValidMoves(current_vector)

        for vector, action in list_validVectors:
            #check if the action is in the children or not 
            (action_num, action_direction) = action 
            count = list_preAction.count(action)
            if count == 0: 
                lower = vector[action_num-1,0]
                upper = vector[action_num-1,1]
                value = upper - lower 
                if value < best_value: 
                    best_action = action 
                    best_vector = vector
        #fully  expanded 
        if not best_action:
            return None  
        matrix = nl.convertToMatrix(best_vector) 
        new_state = State(matrix, state, best_action)
        return new_state





    def best_child(self,state):

    def default_policy(self,state):
    def backup(self,state):
    def __init__(self, function, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    

    def getActionProb(self, currentInput_box, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        reward = 0
        for i in range(self.args.numMCTSSims):
            reward += self.search(currentInput_box)

        s = self.game.stringRepresentation(currentInput_box)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs,reward

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs,reward


    def search(self, currentInput_box):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = self.game.stringRepresentation(currentInput_box)
        board = self.game.getBoardFromInput_box(currentInput_box)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(currentInput_box, THRESHOLD)

        if self.Es[s]!=0:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(board)
            valids = self.game.getValidMoves(currentInput_box, THRESHOLD)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]

        #print(valids)

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        currentInput_box= self.game.getNextState(currentInput_box, a)

        v = self.search(currentInput_box)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v
