import random
import copy
import sys
import operator
import numpy as np


"""
This file has all of the functions for the reinforcement learning algorithm.
Upon initialization, the state space is created and trimmed to exclude impossible
states.

ACTIONS: 0: Load from queue to D2
         1: Move from D2 to PD
         2: Move from PD to D2
         3: Move from D2 to PreR
         4: Move from PreR to ED
         5: Move from ED to PosR
         6: Move from PosR to I
         7: Move from I to DD
         8: Move from DD to O
         9: Move from O to Unload
         10: Remain idle
         11: Gripper rinse sequence
"""
# model params
np.random.seed(0)
policy = None
discount = 0.2
learning_rate = 0.2
exploration_rate = 0.2
terminal_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
rewards = []
rewards_out = []

action_movements = {0: [0, 2],
                    1: [2, 1],
                    2: [1, 2],
                    3: [2, 3],
                    4: [3, 4],
                    5: [4, 5],
                    6: [5, 6],
                    7: [6, 7],
                    8: [7, 8],
                    9: [8, 9],
                    10: [-1, -1],
                    11: [-2, -1]}

S = np.array(np.meshgrid([i for i in range(0, 2)], # queue
                         [i for i in range(0, 2)], # penetrant dip
                         [i for i in range(0, 3)], # dwell
                         [i for i in range(0, 3)], # pre rinse
                         [i for i in range(0, 2)], # emulsify dip
                         [i for i in range(0, 3)], # post rinse
                         [i for i in range(0, 2)], # inspection
                         [i for i in range(0, 2)], # developer dip
                         [i for i in range(0, 3)], # oven
                         [i for i in range(0, 2)], # unload
                         [i for i in range(0, 2)], # redip needed
                         [i for i in range(0, 2)], # 4 hour reached
                         [i for i in range(0, 2)], # oven needed
                         [i for i in range(0, 2)], # first dip needed
                         [i for i in range(0, 2)])).T.reshape(-1, 15) # rinse sequence needed
delete_rows = []
for index in range(0, S.shape[0]):
    row = S[index]
    # can trim state space by enforcing space constraints
    if row[4] + sum(row[6:8]) > 1 or row[4] + sum(row[6:9]) > 3:
        delete_rows.append(index)
    if row[1] == 1 and row[4] + sum(row[6:8]) > 0:
        delete_rows.append(index)
S = np.delete(S, delete_rows, 0)
Q_sa = np.zeros((S.shape[0], 12))
Q_sa.fill(0.25)

def get_new_state(state, action):
    """
    This function takes input of state and action and returns the resulting
    state that would follow from that action. This function also includes
    trimming some states in the form of backtracking, since certain action-state
    pairs can lead to infeasibility
    """
    # current_state is the potential state that would occur from performing the given action
    current_state = np.array(state)

    if not action_movements[action][0] == -1:
        # move blade from start point to finish point based on the action
        if not action_movements[action][0] == -2:
            current_state[action_movements[action][0]] -= 1
            current_state[action_movements[action][1]] += 1
        # change indicator vars
        if current_state[-5] and action == 1: # redip needed and dip action
            current_state[-5] = 0
        if current_state[-4] and action == 3: # advance needed and advance action
            current_state[-4] = 0
        if current_state[-3] and action == 9: # oven needed and unload action
            current_state[-3] = 0
        if current_state[-2] and action == 1: # first dip needed and dip action
            current_state[-2] = 0
        if current_state[-1] and action == 11:
            current_state[-1] = 0

        # determine if the potential state is actually possible. If it is, len will be > 0
        new_state = np.where((S==tuple(current_state)).all(axis=1))[0]
        if len(new_state) > 0:
            # for each of these cases, it would not possible to reach the terminal state
            # to account for this, a negative reward is given and backtracks one step
            # trying to advance before 4 hours has been reached
            if action == 3 and not state[-4]:
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            # trying to dip when nothing needs to be dipped
            elif action == 1 and state[-2] == 0 and state[-5] == 0:
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            # trying to remove from oven when not yet done
            elif action == 9 and state[-3] == 0:
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            # if a blade need redip but the action is to advance
            elif state[-5] and action != 1 and not state[1] and state[4] + sum(state[6:8]) == 0:
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            # if oven needed but action is to do something other than required redip
            elif state[-3] and (action == 3 or action == 4 or action == 0):
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            elif state[-3] and state[-2] and not state[-5] and action == 1:
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            # if trying to move from prerinse but post rinse occupied
            elif sum(state[4:8]) > 0 and action == 4:
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            # if violating any of the hold in place rules
            elif (state[4] and action != 5) or (state[6] and action != 7) or (state[7] and action != 8) or (state[1] and action != 2):
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            # trying to move to the oven when there is no space
            elif action == 6 and (state[8] == 2 or state[-3]):
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            elif state[-1] and action not in [1, 2, 11, 0]:
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            elif not state[-1] and action == 11:
                return np.where((S==tuple(state)).all(axis=1))[0][0]
            else:
                return new_state[0]
        else:
            return np.where((S==tuple(state)).all(axis=1))[0][0]
    else:
        return np.where((S==tuple(state)).all(axis=1))[0][0]

def update_value(state, action, reward, test):
    """
    function for updating the state_action pair values based on the action
    taken and the reward received. Test is a boolean indicating whether the
    program is being trained or tested.
    """
    global rewards
    rewards.append(reward)
    # print(sum(rewards))
    new_state = get_new_state(state, action)
    state_index = np.where((S==tuple(state)).all(axis=1))[0][0]
    if not test:
        Q_sa[state_index][action] += learning_rate * (reward + discount * np.max(Q_sa[new_state][:]) - Q_sa[state_index][action])
    return False if new_state == state_index else True

def get_action(state):
    # function returns the greedy choice for best action given the state or
    # a random action according to some probability specified in the params
    if random.uniform(0, 1) < exploration_rate:
        next_action = random.randint(0, 10)
    else:
        next_action = np.argmax(Q_sa[state][:])

    return next_action

def decision(state_array, test=False):
    """
    function makes the decision based on the state space array.
    this will either be an index lookup on the policy if the algorithm is in
    test mode (and a policy is supplied) or it will make a call to get_action()
    """
    if np.array_equal(np.array(state_array), terminal_state):
        # print(sum(rewards))
        return None, None, None
    state_index = np.where((S==tuple(state_array)).all(axis=1))[0][0]
    action = get_action(state_index)
    if test:
        action = policy[state_index]
    return action, action_movements[action][1], action_movements[action]

def save_policy(i):
    global policy
    if not str(i) == 'est':
        policy = np.argmax(Q_sa, axis=1)
    np.savetxt('policy_' + str(i) + '.out', policy, delimiter=',')

def save_rewards():
    global rewards_out
    np.savetxt('rewards.out', np.array(rewards_out), delimiter=',')

def average_rewards():
    global rewards, rewards_out
    rewards_out.append(sum(rewards)/len(rewards))
    rewards = []


def get_policy():
    return np.argmax(Q_sa, axis=1)

def save_values():
    np.save('values.npy', Q_sa)

def load_values(val):
    global Q_sa
    Q_sa = np.load(val)

def load_policy(pol):
    global policy
    policy = np.loadtxt(pol, delimiter=',')

def estimate_unvisited():
    # print('\n\nCalculating estimates for unvisited states...')
    global policy, S, Q_sa
    index = 0
    for row in Q_sa:
        sys.stdout.write('Estimation completion: %f percent\r' % ( 100 * index / len(S)))
        sys.stdout.flush()
        # find row where all values = 0.25 (iff has not been visited)
        if np.sum(row) == len(row) * 0.25:
            vote_count = dict(zip(range(0,12), [0] * 12))
            state = S[index]
            for j in range(0, len(state)):

                temp = copy.copy(state)
                if state[j] < 2:
                    temp[j] = state[j] + 1
                    # find index of temp
                    new_state = np.where((S==tuple(temp)).all(axis=1))[0]
                    # get action from policy. But not if temp is also all 0.25
                    if len(new_state) > 0 and np.sum(Q_sa[new_state[0]]) != len(row) * 0.25:
                        vote_count[policy[new_state[0]]] += 1

                temp = copy.copy(state)
                if state[j] > 0:
                    temp[j] = state[j] - 1
                    # find index of temp
                    new_state = np.where((S==tuple(temp)).all(axis=1))[0]
                    # get action from policy. But not if temp is also all 0.25
                    if len(new_state) > 0 and np.sum(Q_sa[new_state[0]]) != len(row) * 0.25:
                        vote_count[policy[new_state[0]]] += 1
            # set action for unvisited state as highest vote count from neighbors
            policy[index] = max(vote_count, key=vote_count.get)
            # print('\n' + str(index))
            # print(state)
            # print(vote_count)
            # print(policy[index])
            # input(' ')
        index += 1
