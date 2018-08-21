from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from torch.distributions import Categorical

import matplotlib.pyplot as plt

THRESHOLD = 0.01
class ActorCritic():

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()


    def executeEpisode(self):

        currentInput_box = self.game.input_box
        board = self.game.getBoardFromInput_box(currentInput_box)

        episodeStep = 0

        while True:
            episodeStep += 1

            example = []

            pi, v = self.nnet.predict(board)

            example.append(board)

            valids = self.game.getValidMoves(currentInput_box, THRESHOLD)

            p=(pi * valids)
            p = p/np.sum(p)

            action = np.random.choice(len(pi), p=p)

            example.append(action)

            currentInput_box = self.game.getNextState(currentInput_box, action)


            r = self.game.getGameEnded(currentInput_box, THRESHOLD)

            if r == -3:
                board = self.game.getBoardFromInput_box(currentInput_box)
                _, v_prime = self.nnet.predict(board)
                delta = 0.8 * v_prime - v

                example.append(delta)

                self.nnet.train(example)

            else:
                example.append(r)

                self.nnet.train(example)

                return r, episodeStep

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                reward_list = []
                count_list = []
                step_list = []

                for eps in range(self.args.numEps):

                    reward, step_count = self.executeEpisode()

                    step_list.append(step_count)
                    reward_list.append(reward)
                    count_list.append(eps)

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                plt.scatter(count_list, reward_list, label = 'rewards_training')
                plt.savefig("fig/rewards_"+str(i)+".png")
                plt.close()
                plt.scatter(count_list, step_list, label = 'steps_training')
                plt.savefig("fig/steps_"+str(i)+".png")
                plt.close()

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
