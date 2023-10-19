#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################

    greedy_action_idx = np.argmax(Q[state])
    prob = np.ones(nA) * epsilon / nA  
    prob[greedy_action_idx] += (1 - epsilon)  
    action = np.random.choice(np.arange(nA), p=prob) # choose action based on prob
    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    '''20 points'''
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    n_action = env.action_space.n

    for _ in range(n_episodes):
        epsilon = 0.99 * epsilon 
        curr_state = env.reset()  # initialize the environment
        # print(curr_state)
        # print(curr_state[0])
        curr_state = curr_state[0]
        curr_action = epsilon_greedy(Q, curr_state, n_action, epsilon)
        done = False
        while not done: 
            next_state, reward, done, _, _ = env.step(curr_action) 
            next_action = epsilon_greedy(Q, next_state, n_action, epsilon)  
            # TD update
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[curr_state][curr_action]
            Q[curr_state][curr_action] += alpha * td_error

            curr_state = next_state  # update state
            curr_action = next_action  # update action
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    '''20 points'''
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    n_action = env.action_space.n

    for _ in range(n_episodes):
        epsilon = 0.99 * epsilon  # define decaying epsilon
        curr_state = env.reset()  # initialize the environment
        curr_state = curr_state[0]
        done = False
        while not done:  
            curr_action = epsilon_greedy(Q, curr_state, n_action, epsilon)  
            next_state, reward, done,_,_ = env.step(curr_action)

            # TD update
            best_action = np.argmax(Q[next_state])
            td_target = reward + gamma*Q[next_state][best_action]  # td_target with best Q
            td_error = td_target - Q[curr_state][curr_action]  # td_error

            Q[curr_state][curr_action] += alpha*td_error  # new Q

            curr_state = next_state  # update state
    ############################
    return Q
