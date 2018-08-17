import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class NaiveNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()    # self.board_x is the number of variables
                                                            # self.board_y is the size of the embedding
        self.action_size = game.getActionSize()
        self.args = args

        super(NaiveNNet, self).__init__()


        self.fc1 = nn.Linear(self.board_y, 32)


        self.fc2 = nn.Linear(32, 64)

        self.fc3 = nn.Linear(64, 16)

        self.fc4 = nn.Linear(16, 2)

    def forward(self, s):

        s = s.view(-1, self.board_x, self.board_y)



        s = F.sigmoid(self.fc1(s))

        s = F.sigmoid(self.fc2(s))

        #print(s)

        s = self.fc3(s)

        pi = self.fc4(s)

        pi = pi.view(-1, self.action_size)


        return F.softmax(pi, dim=1)
