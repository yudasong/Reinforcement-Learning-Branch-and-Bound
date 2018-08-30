8/30
1. Add new feature: estimation of a lower value.

8/17
1. try randomly secting the domain. 
2. Using the same network, can we generalize f(x) = 0 to f(x) = n?
3. Sample from the neiborhood of every point.
4. Compare with branch and prune, even the network doesn't give a fair reward.\
    a. train f(x) = 0\
    b. compare B&P f(x) = 0 and B&P+NN f(x) = 0\
    c. compare B&P f(x) = n and B&P+NN f(x) = n

8/10

1. converges after finding a fair enough solution.
2. nn fails to find answer if the domain is changed after training.

8/9

Can't use tilted value cuz the weights for policy and value are shared.
When the interval is small, branching intervals will lead to the same action again, until masked. (Because the input for NN is very similar).

8/3/2018

1. Try eliminating value head.
2. Try pure MCTS. 
3. Try difficult functions.


7/24/2018
1. multi dimension issue within sampling data representation
2. Need unified naming among BB files
3. Need benchmarks
4. Passing messages among nodes in a graph: https://arxiv.org/pdf/1704.01212.pdf
5. Reward calculation. Currently by 1-|value in function| and collect all terminal reward as training example
6. backtrack


7/13/2018
1. We need a systemetic guide/ introduction to branch and bound.
2. We need a systemetic guide/ introduction to branch and bound.
3. How to choose the middle value to cut? Why 0.4 ?
4. Does relaxation mean making problem easier?
5. what does subproblem infeasible mean? 
6. Documentation for pyibex.
7. Representation of the value, can it be all nagative?
8. Representation of the state.
9. When it reaches ternimal, how to calculate its reward? Possible approach: calculate the mean give the lower and upper value.

