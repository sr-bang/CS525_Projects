#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################

    score = observation[0] # player score
    # print(observation)
    # print(score)
    if(score >= 20):
        action = 0
    else:
        action = 1
    ############################
    return action


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################

    for _ in range(n_episodes):
        episode = []     # initialize a new episode
        observation = env.reset()
        observation = observation[0] # get the first element of the tuple
        while True:
            # print(len(observation))
            action = policy(observation)
            next_state, reward, done,_, _ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                break
            observation = next_state
        G_state = [] # G for each state
        G = 0
        for (state, action, reward) in reversed(episode):
            G = gamma * G + reward
            G_state.append(G)
        G_state.reverse() # order of G_state is the same as episode

        # Find unique states in the episode
        visited_states = []

        for i in range(len(episode)):
            state = episode[i][0]
            if state in visited_states:
                continue
            visited_states.append(state) # add state to visited_states

            returns_sum[state] += G_state[i]
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]

    ############################

    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: 
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
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #

    greedy_action_idx = np.argmax(Q[state])
    prob = np.ones(nA) * epsilon / nA  
    prob[greedy_action_idx] += (1 - epsilon)  # greedy action: higher probability
    action = np.random.choice(np.arange(nA), p=prob)

    ############################
    return action


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # while epsilon > 0:
    for _ in range(n_episodes):
        observation = env.reset()
        observation = observation[0] # get the first element of the tuple
        episode = []
        while True:
            action = epsilon_greedy(Q, observation, env.action_space.n, epsilon)
            new_state, reward, done, _,_ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                break
            observation = new_state

        G_state = [] # G for each state
        G = 0
        for (state, action, reward) in reversed(episode):
            G = gamma * G + reward
            G_state.append(G)
        G_state.reverse()

        # For all states in the episode
        visited = []
        for i in range(len(episode)):
            state, action = episode[i][0], episode[i][1]
            if (state, action) in visited:
                continue
            visited.append((state, action)) # add state to visited_states

            returns_sum[(state, action)] += G_state[i]
            returns_count[(state, action)] += 1
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

        epsilon -= 0.1/n_episodes # decay epsilon

    ############################

    return Q
