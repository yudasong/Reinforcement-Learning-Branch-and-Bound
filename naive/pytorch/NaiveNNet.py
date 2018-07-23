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
        


        self.fc1 = nn.Linear(self.board_x * self.board_y, 32)
        self.fc_bn1 = nn.BatchNorm1d(32)

        self.fc2 = nn.Linear(32, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, self.action_size)

        self.fc4 = nn.Linear(128, 32)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        
        s = F.relu(self.fc_bn1(self.fc1(s)))
        s = F.relu(self.fc_bn2(self.fc2(s)))

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 32
        v = self.fc5(v)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), F.tanh(v)
