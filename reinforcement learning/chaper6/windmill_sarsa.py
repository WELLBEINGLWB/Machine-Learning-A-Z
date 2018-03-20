import numpy as np
import random
import matplotlib.pyplot as plt

# SOLVED
# Field
HEIGHT = 7
WIDTH = 10
# start position
START_I = 3
START_J = 0
# goal position
GOAL_I = 3
GOAL_J = 7
# undiscounted case
DISCOUNT = 1
REWARD = -1
# step size parameter
ALPHA = 0.5
# how many total steps are taken for estimating Q_s_a
STEPS_LIM = 200000

actions = dict()
actions['UP'] = (-1,0)
actions['DOWN'] = (1,0)
actions['LEFT'] = (0,-1)
actions['RIGHT'] = (0,1)

Q_s_a = []
#states_actions = []
for i in range(HEIGHT):
    for j in range(WIDTH):
        for k in actions:
#            states_actions.append(((i,j), k))
            if i == GOAL_I and j == GOAL_J:
#                Q_s_a.append(((i,j), k, 0.0))
#                Q_s_a[i][j][k] = 0.0
                Q_s_a.append(((i, j), k, 0.0))
            else:
                Q_s_a.append(((i, j), k, np.random.rand()*2-1))
#                Q_s_a[i][j][k] = np.random.rand()*2-1
#                Q_s_a.append(((i,j), k, np.random.rand()*2-1))
del i, j , k

episodes_counter = 0
steps_counter = 0
steps_episodes = []
def play_one_episode(Q_s_a, verbose=True):
    global episodes_counter
    global steps_counter
    global steps_episodes
    episodes_counter += 1

    s = (START_I, START_J)
    s_ = None
    a = None
    a_ = None

    while True:
        steps_counter += 1
        # choose action from state s
        if np.random.rand() < 1/episodes_counter:
            # return random action
            a = random.choice(list(actions))
            if verbose == True:
                print("a chosen randomly")
        else:
            bestValue = -float('Inf')
            for l in range(len(Q_s_a)):
                if Q_s_a[l][0] == s and Q_s_a[l][2] > bestValue:
                    bestValue = Q_s_a[l][2]
                    a = Q_s_a[l][1]

        if verbose == True:
            if a == None:
                print("ERROR: a not assigned")

        # calculate new state s_
        s_ = (s[0]+actions[a][0], s[1]+actions[a][1])
        # apply wind to new state
        if (s[1] >= 3 and s[1] <= 5) or s[1] == 8:
            s_ = (s_[0] - 1 , s_[1])
        elif s[1] == 6 or s[1] == 7:
            s_ = (s_[0] - 2, s_[1])

        # make sure state is on the field
        if s_[0] < 0:
            s_ = (0, s_[1])
        elif s_[0] > HEIGHT-1:
            s_ = (HEIGHT-1, s_[1])
        elif s_[1] < 0:
            s_ = (s_[0], 0)
        elif s_[1] > WIDTH-1:
            s_ = (s_[0], WIDTH-1)



        # choose action a_ depending on epsilon greedy strategy
        if np.random.rand() < 1/episodes_counter:
            # return random action
            a_ = random.choice(list(actions))
            if verbose == True:
                print("a_ chosen randomly")
        else:
            bestValue = -float('Inf')
            for l in range(len(Q_s_a)):
                if Q_s_a[l][0] == s and Q_s_a[l][2] > bestValue:
                    bestValue = Q_s_a[l][2]
                    a_ = Q_s_a[l][1]

        if verbose == True:
            if a_ == None:
                print("ERROR: a_ not assigned")

        # update Q_s_a with (s, a, r, s_, a_)
        for l in range(len(Q_s_a)):
            if Q_s_a[l][0] == s and Q_s_a[l][1] == a:
                for m in range(len(Q_s_a)):
                    if Q_s_a[m][0] == s_ and Q_s_a[m][1] == a_:
                        newQValue = Q_s_a[l][2] + ALPHA*(REWARD + DISCOUNT * Q_s_a[m][2] - Q_s_a[l][2])
                        Q_s_a[l] = (s, a, newQValue)
#                        Q_s_a[l][2] = Q_s_a[l][2] + ALPHA*(REWARD + DISCOUNT * Q_s_a[m][2] - Q_s_a[l][2])

        # update s, and a
        s = s_
        a = a_

        # check for terminal state
        if s_ == (GOAL_I, GOAL_J):
            print("TERMINAL STATE")
            steps_episodes.append((steps_counter, episodes_counter))
            return False
        elif steps_counter > STEPS_LIM:
            print("STEPSCOUNTER REACHED")
            steps_episodes.append((steps_counter, episodes_counter))
            return True


while True:
    if play_one_episode(Q_s_a) == True:
        break

plt.figure(1)
x_plot = [x[0] for x in steps_episodes]
y_plot = [x[1] for x in steps_episodes]
plt.scatter(x_plot, y_plot)
plt.show()







