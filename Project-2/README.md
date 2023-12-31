# DS551/CS525 2023 Fall Individual Project 2
# Model-free Algorithms (Monte-Carlo and Temporal Difference)

## Goals
In this project, you will be asked to implement two model-free algorithms. The first one is Monte-Carlo(MC), including  the first visit of on-policy MC prediction and on-policy MC control for [blackjack](https://gym.openai.com/envs/Blackjack-v0/). The second one is Temporal-Difference(TD), including Sarsa(on-policy) and Q-Learning(off-policy) for [cliffwalking](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py).

<img src="/Project-2/img/project2-1updated.png" width="80%" >
<img src="/Project-2/img/project2-2updated.png" width="80%" >

## Guidelines
* Implement functions in mc.py and td.py
* Evaluate functions by typing "nosetests -v mc_test.py" and "nosetests -v td_test.py" respectively, or try "python3 -m nose -v mc_test.py" and "python3 -m nose -v td_test.py". (put your function file and test file into the same folder).

## Hints
* On-policy first visit Monte-Carlo prediction
<img src="/Project-2/img/mc_predict.png" width="80%" >

* On-policy first visit Monte-Carlo control
<img src="/Project-2/img/mc.png" width="80%" >

* Sarsa (on-policy TD control)
<img src="/Project-2/img/sarsa.png" width="80%" >

* Q-learing (off-policy TD control)
<img src="/Project-2/img/q-learning.png" width="80%" >

## Results

#### Result for Monte-Carlo(MC)
<img src="/Project-2/img/result1.png" width="80%" >

#### Result for Temporal-Difference(TD)
<img src="/Project-2/img/result2.png" width="80%" >

