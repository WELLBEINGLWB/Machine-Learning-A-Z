#
from __future__ import print_function
import numpy as np
import random

R_LOSS = -1
R_DRAW = 0
R_WIN = 1
DELTA = 1e-20
# create states array
# states[i][0] = players sum of cards
# states[i][1] = usable ace
# states[i][2] = dealer's card
states = []

for cardSum in range(12, 22):
    for usableAce in range(2):
        for dealerCard in range(1, 11):
            triple = cardSum, usableAce, dealerCard
            states.append(triple)

# create initial policy sumCards > 20, stick, else hit
# 0 = stick, 1 = hit
policy = np.zeros(len(states), int)
for i in range(len(states)):
    if states[i][0] >= 20:
        policy[i] = 0
    else:
        policy[i] = 1

# create initial state action value table
# create states array
# Q[i][0] = players sum of cards
# Q[i][1] = usable ace
# Q[i][2] = dealer's card
# Q[i][3] = stick/hit
Q = []
for cardSum in range(12, 22):
    for usableAce in range(2):
        for dealerCard in range(1,11):
            for action in range(2):
                triple = cardSum, usableAce, dealerCard
                Q.append((triple, action))
del dealerCard, cardSum, usableAce, action, triple

Q_s_a = np.zeros(len(Q))

# create Returns list for each state-action pair
returns_s_a = [[] for i in range(len(Q))]; del i

def get_card():
    card = random.randint(1, 13)
    card = min(card, 10)
    return card

# idea equal probability of getting each card
def play_one_episode(states, policy, exploringStart=True, verbose=False):

    initStateInd = random.randint(0, len(states) - 1)
    playerSum = states[initStateInd][0]
    usableAce = states[initStateInd][1]
    dealerSum = states[initStateInd][2]

#    action = policy[initStateInd]
    trajectory = []
#    trajectory.append(initState, action)

    # players choices
    while True:
        if exploringStart == True:
            action = random.randint(0,1)
            exploringStart = False
        else:
            action = policy[states.index((playerSum, usableAce, dealerSum))]

        trajectory.append(((playerSum, usableAce, dealerSum), int(action)))

        if action == 0:
            break

        card = get_card()
        playerSum += card

        if (playerSum > 21) and (usableAce == 1):
            # use the ace as 1
            playerSum -= 10
            usableAce = 0
        elif (playerSum > 21) and (usableAce == 0):
            if verbose == True:
                print("LOSS ---- PLAYERSUM = ", playerSum)
            return R_LOSS, trajectory

    # prepare dealers starting hand
    dealerCard2 = get_card()
    if dealerSum == 1 and dealerCard2 != 1:
        dealerUsableAce = True
        dealerSum += 10 + dealerCard2
    elif dealerSum == 1 and dealerCard2 == 1:
        dealerUsableAce = True
        dealerSum += 11
    elif dealerSum != 1 and dealerCard2 == 11:
        dealerUsableAce = True
        dealerSum += 11
    else:
        dealerSum += dealerCard2
        dealerUsableAce = False

    # dealers turn
    while True:
        if dealerSum > 17:
            dealerAction = 0
        else:
            dealerAction = 1

        if dealerAction == 1:
            dealerSum += get_card()
        else :
            break

        if (dealerSum > 21) and (dealerUsableAce == True):
            dealerSum -= 10
            dealerUsableAce = False
        elif (dealerSum > 21) and (dealerUsableAce == False):
            if verbose == True:
                print("WIN ---- DEALERSUM = ", dealerSum)
            return R_WIN, trajectory

    if verbose == True:
        print("RESULT ---- PLAYER = ", playerSum, "DEALER = ", dealerSum)
    # compare sum of players
    if playerSum == dealerSum :
        return R_DRAW, trajectory
    elif playerSum > dealerSum:
        return R_WIN, trajectory
    elif playerSum < dealerSum:
        return R_LOSS, trajectory
# %%
#plays_counter = 0
#while True:
#    print(plays_counter)
#    plays_counter += 1
#    play_one_episode(states, policy)

# %%
PLAYS = 200000

for i in range(PLAYS):
    r, traj = play_one_episode(states, policy)

    for state, action in traj:
        state_action_index = Q.index((state, action))
        returns_s_a[state_action_index].append(r)
        # calculate average returns for state_action pair
        # makre sure division by 0 does not take place
        Q_s_a[state_action_index] = sum(returns_s_a[state_action_index]) / \
                                len(returns_s_a[state_action_index])

#     policy update
    for state, action in traj:
        state_action_index = Q.index((state, action))
        if state_action_index % 2 == 1:
            state_action_index -= 1
        bestAction = None
        # stick best option
        if Q_s_a[state_action_index] > Q_s_a[state_action_index+1]:
            bestAction = 0
        # else hit is best option
        elif Q_s_a[state_action_index] < Q_s_a[state_action_index+1]:
            bestAction = 1

        if bestAction == None:
            continue
        elif action != bestAction:
            stateIndex = states.index(state)
            print(state, int(action), "new action ",bestAction)
            policy[stateIndex] = bestAction
#
#
policyUsableAce = []
for i in range(len(states)):
    if states[i][1] == 1:
        policyUsableAce.append((states[i], policy[i]))

policyNoAce = []
for i in range(len(states)):
    if states[i][1] == 0:
        policyNoAce.append((states[i], policy[i]))

    # policy update
#    for state, action in traj:
#        state_action_index = Q.index((state, action))
#        state_index = states.index(state)
#        if state_action_index % 2 == 1:

#            state_action_index -= 1
#
#        best_action_state_action_index = np.argmax(Q_s_a[state_action_index:state_action_index+2])
#        policy[state_index] = best_action_state_action_index
#
#
## print policy
#states_policy_list = []
#for state in range(len(states)):
#    if states[state][1] == 1:
#        states_policy_list.append((states[state], int(policy[state])))



