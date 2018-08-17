import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torchvision import datasets, transforms
from torch.autograd import Variable

from .NaiveNNet import NaiveNNet as nnnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = nnnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):

        optimizer = optim.Adam(self.nnet.parameters())


        boards, actions, rewards = list(zip(*[examples[i] for i in range(len(examples))]))

        for i in range(len(boards)):

            board = torch.FloatTensor(boards[i].astype(np.float64))
            board = Variable(board)
            out_pi = self.nnet(board)
            l_pi = self.loss_pi(rewards[i], actions[i], out_pi)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            l_pi.backward()
            optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = Variable(board, volatile=True)
        board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        pi = self.nnet(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi.data.cpu().numpy()[0]

    def loss_pi(self, reward, action, output):

        output = output[:, action].view(1, -1)
        log_prob = output.log()
        loss = -log_prob * reward

        return loss

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
