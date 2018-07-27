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
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(NaiveNNet, self).__init__()



        self.fc1 = nn.Linear(self.board_y, 32)


        self.fc2 = nn.Linear(32, 128)

        self.fc3 = nn.Linear(128, self.action_size)

        self.fc4 = nn.Linear(128, 16)

        self.fc5 = nn.Linear(32, 1)

    def forward(self, s):
        #
        s = s.view(-1, self.board_x, self.board_y)

        s = F.relu(self.fc1(s))

        s = F.relu(self.fc2(s))

        pi = self.fc3(s)

        v = self.fc4(s)

        v = v.view(-1, 32)

        v = self.fc5(v)
        return F.log_softmax(pi, dim=1)[0][0], F.tanh(v[0])
