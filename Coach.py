from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import matplotlib.pyplot as plt

THRESHOLD = 0.001
class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args, round):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

        self.show = False

        self.round = round

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        currentInput_box = self.game.input_box
        board = self.game.getBoardFromInput_box(currentInput_box)
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            pi, reward = self.mcts.getActionProb(currentInput_box, temp=temp)

            #print(pi)



            trainExamples.append([board, pi])

            # choose the action = argmax from the policy of the nnet
            #action = np.argmax(self.nnet(currentInput_box))


            action = np.random.choice(len(pi), p=pi)
            currentInput_box = self.game.getNextState(currentInput_box, action)
            #currentInput_box = self.game.distortInputbox(currentInput_box)
            board = self.game.getBoardFromInput_box(currentInput_box)
            r = self.game.getGameEnded(currentInput_box, THRESHOLD)

            if r!=0:
                return [(x[0],x[1],r) for x in trainExamples], episodeStep

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

            std = 999
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                reward_list = []
                count_list = []
                step_list = []


                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree

                    example, step_count = self.executeEpisode()
                    iterationTrainExamples += example


                    step_list.append(step_count)
                    reward_list.append(iterationTrainExamples[-1][2])
                    count_list.append(eps)

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()



                plt.scatter(count_list, reward_list, label = 'rewards_training')
                plt.savefig("fig/"+str(self.round)+"_rewards_"+str(i)+".png")
                plt.close()
                plt.scatter(count_list, step_list, label = 'steps_training')
                plt.savefig("fig/"+str(self.round)+"_steps_"+str(i)+".png")
                plt.close()

                iterationTrainExamples, std, mean = self.normalizeReward(iterationTrainExamples)

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            #self.saveTrainExamples(i-1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            self.show = True
            nmcts = MCTS(self.game, self.nnet, self.args)

            """

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')"""
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            if std < 20 and mean < 0:
                print("stop traing because of identical rewards")
                break



    def normalizeReward(self, examples):
        rewards = []
        cur = 0
        result = []
        for x in examples:
            if x[-1] != cur:
                rewards.append(x[-1])
                cur = x[-1]

        min = np.min(np.asarray(rewards))
        max = np.max(np.asarray(rewards))
        for i in range(len(examples)):
            if examples[i][-1] == 1000:
                reward = -1
                result.append((examples[i][0], examples[i][1], reward))
            elif examples[i][-1] < 0:
                reward= examples[i][-1] / min
                result.append((examples[i][0], examples[i][1], reward))
            else:
                reward= -examples[i][-1] / max
                result.append((examples[i][0], examples[i][1], reward))

        return result, np.std(np.asarray(rewards)), np.mean(np.asarray(rewards))

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
