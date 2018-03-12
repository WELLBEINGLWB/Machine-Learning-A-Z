from __future__ import print_function
import numpy as np

WORLD_SIZE = 4
REWARD = -1
ACTION_PROB = 0.25

world = np.zeros((WORLD_SIZE, WORLD_SIZE))

actions = ['L', 'U', 'R', 'D']

next_state = []
for i in range(WORLD_SIZE):
    next_state.append([])
    for j in range(WORLD_SIZE):
        next_state_entry = dict()

        if i == 0:
            next_state_entry['U'] = [i, j]
        else:
            next_state_entry['U'] = [i - 1, j]

        if i == WORLD_SIZE - 1:
            next_state_entry['D'] = [i, j]
        else :
            next_state_entry['D'] = [i + 1, j]

        if j == 0:
            next_state_entry['L'] = [i, j]
        else:
            next_state_entry['L'] = [i, j - 1]

        if j == WORLD_SIZE - 1:
            next_state_entry['R'] = [i, j]
        else:
            next_state_entry['R'] = [i, j + 1]

        next_state[i].append(next_state_entry)

states = []
for i in range(WORLD_SIZE):
    for j in range(WORLD_SIZE):
        if (i == 0 and j == 0) or (i == WORLD_SIZE - 1 and j == WORLD_SIZE - 1):
            continue
        else:
            states.append([i, j])

while True:
    new_world = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i, j in states:
        for a in actions:
            new_position = next_state[i][j][a]
            # bellman
            new_world[i, j] += ACTION_PROB * (REWARD + world[new_position[0], new_position[1]])
    if np.sum(np.abs(world - new_world)) < 1e-2:
        print("RANDOM POLICY")
        print(new_world)
        break
    world = new_world.copy()