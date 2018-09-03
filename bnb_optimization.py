from __future__ import print_function
from pyibex import *
from generateFunctions import generateFunctions
from GFwithMonomial import GFwithMonomial as GF

import numpy as np

from naive.BB import BB
from naive.pytorch.NNet import NNetWrapper as nn

from heapq import *

import matplotlib.pyplot as plt


def get_lb(f, input_box):
    b = f.eval(input_box)
    return b.lb()

def eval_at_mid(f, input_box):
    #ibex documentation is offline so just doing this stupidly
    b = IntervalVector(input_box.mid())
    return f.eval(b).mid()

def minimize(f, input_box, eps):
    step = 0
    box_stack = [input_box]
    g_min = eval_at_mid(f, input_box)
    while box_stack:
        b = box_stack.pop()
        if g_min < get_lb(f, b):
            continue
        for i in range(len(b)):
            if b[i].diam() > 0.001:
                new_boxes = b.bisect(i, 0.5)
                for i in range(2):
                    #if a sample is lower, update global min to that
                    if eval_at_mid(f, new_boxes[i]) < g_min - eps:
                        g_min = eval_at_mid(f, new_boxes[i])
                    #only push the box if its overestimation of lower bound is lower
                    if get_lb(f, new_boxes[i]) < g_min - eps:
                        box_stack.append(new_boxes[i])
                        step += 1
                break
    return g_min, step

def min_with_nn(f, input_box, esp, func):
    game = BB(f, input_box, Interval(-999,999), func)
    nnet = nn(game)

    nnet.load_checkpoint('./ckpsss/', 'best.pth.tar')

    current_box = input_box
    board = game.getBoardFromInput_box(current_box)



    r = game.getGameEnded(current_box, esp)
    step = 0
    while r == 0:
        board = game.getBoardFromInput_box(current_box)

        #print(board)

        pi, v = nnet.predict(board)
        pi = game.getValidMoves(current_box, esp) * pi
        print(pi)
        a = np.argmax(pi)
        print(a)
        current_box = game.getNextState(current_box, a)
        r = game.getGameEnded(current_box, esp)
        step += 1

    return r, step

def bnb_with_nn_pq(f, input_box, eps, func):

    QSA = {}


    game = BB(f, input_box, Interval(-999,999), func)
    nnet = nn(game)

    nnet.load_checkpoint('./ckps/', 'best.pth.tar')

    current_box = input_box
    board = game.getBoardFromInput_box(current_box)

    pi, v = nnet.predict(board)

    box_stack =  [(v, current_box)]

    QSA[game.stringRepresentation(current_box)] = (pi, v)

    g_min = eval_at_mid(f, current_box)

    step = 0
    while box_stack:

        print(box_stack)

        v, current_box = heappop(box_stack)

        s = game.stringRepresentation(current_box)

        pi, v = QSA[s]

        pi = game.getValidMoves(current_box, eps) * pi

        print(pi)

        a = np.argmax(pi)

        pi[a] = 0

        QSA[s] = (pi, v)

        if np.sum(pi) > 0 and get_lb(f, current_box) < g_min - eps:

            heappush(box_stack, (v, current_box))

        current_box = game.getNextState(current_box, a)

        if eval_at_mid(f, current_box) < g_min - eps:
            g_min = eval_at_mid(f, current_box)
        #only push the box if its overestimation of lower bound is lower
        if get_lb(f, current_box) < g_min - eps:

            board = game.getBoardFromInput_box(current_box)


            pi, v = nnet.predict(board)

            heappush(box_stack, (v, current_box))
            QSA[game.stringRepresentation(current_box)] = (pi, v)

        step += 1


    return g_min, step

def bnb_with_nn(f, input_box, eps, func):

    QSA = {}
    game = BB(f, input_box, Interval(-999,999), func)
    nnet = nn(game)
    nnet.load_checkpoint('./ckpsss/', 'best.pth.tar')

    current_box = input_box
    board = game.getBoardFromInput_box(current_box)
    pi, v = nnet.predict(board)
    box_stack =  [current_box]
    QSA[game.stringRepresentation(current_box)] = (pi, v)
    g_min = eval_at_mid(f, current_box)
    step = 0

    while box_stack:

        current_box = box_stack.pop()

        if g_min < get_lb(f, current_box):
            continue

        s = game.stringRepresentation(current_box)

        pi, v = QSA[s]

        pi = game.getValidMoves(current_box, eps) * pi

        if np.sum(pi) > 0:

            a = np.argmax(pi)
            if pi[a] > 0:
                pi[a] = 0
                QSA[s] = (pi, v)

                if a % 2 == 0:
                    alter_a = a + 1
                else:
                    alter_a = a - 1

                alter_box = game.getNextState(current_box, alter_a)

                current_box = game.getNextState(current_box, a)

                new_boxes = [alter_box, current_box]

                for b in new_boxes:
                    if eval_at_mid(f, b) < g_min - eps:
                        g_min = eval_at_mid(f, b)
                        #only push the box if its overestimation of lower bound is lower
                    if get_lb(f, b) < g_min - eps:

                        s = game.stringRepresentation(b)
                        if s not in QSA:
                            board = game.getBoardFromInput_box(b)
                            pi, v = nnet.predict(board)
                            QSA[s] = (pi, v)
                        if b not in box_stack:
                            box_stack.append(b)
                            step += 1
    return g_min, step

bb_list = []
nn_list = []
bb_sum = 0
nn_sum = 0
bb_result = []
nn_result = []
count = []

for i in range(1,101):
    #Original function
    generator = GF(["x1","x2"],[3,3],-5,5,20)
    generator.randomPara()
    function = generator.generateString(generator.coe, generator.degree_matrix)
    func = generator.generateFunc
    f = Function("x1","x2", function)
    #Define the input domain of the function
    input_box = IntervalVector([[-5,5],[-5,5]])
    min, step = minimize(f, input_box, 0.001)
    bb_list.append(step)
    bb_sum += step
    bb_result.append(min)
    print("min of ", f, "in range", input_box, ":", min, "\nstep count:", step)
    min, step = min_with_nn(f, input_box, 0.001, func)
    #nn_result.append(min)
    print("min of ", f, "in range", input_box, ":", min, "\nstep count:", step)
    min, step = bnb_with_nn(f, input_box, 0.001, func)
    nn_list.append(step)
    nn_sum += step
    print("min of ", f, "in range", input_box, ":", min, "\nstep count:", step)
    count.append(i)
    plt.scatter(count, bb_list, label = 'bb')
    plt.scatter(count, nn_list, label = 'nn')
    plt.savefig("steps.png")
    plt.close()
    #plt.scatter(count, bb_result, label = 'bb')
    #plt.scatter(count, nn_result, label = 'nn')
    #plt.savefig("results.png")
    #plt.close()

    print("average:","bb:",bb_sum/i, "nn:",nn_sum/i)

plt.scatter(count, bb_list, label = 'bb')
plt.scatter(count, nn_list, label = 'nn')
plt.plot()
