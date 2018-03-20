# Recreate figure 4.1 of Reinforcement learning an introduction
# Dynamic programming
# iterative policy evaluation
import numpy as np

THRESHOLD = 1e-2
REWARD = -1
GAMMA = 1 # undiscounted case

def print_value_table(V):
    for i in range(len(V)):
            for j in range(len(V)):
                if V[i,j] < 0:
                    print("%2.3f   " % V[i,j], end="")
                else:
                    print(" %2.3f   " % V[i,j], end="")
            print("\n")
        
        
def sweep_once(V, V_static, i, j, actions):
    for i in range(height):
        for j in range(width):
            
            V[i,j] = 0
            if i == 0 and j == 0:
                continue
            elif i == height - 1 and j == width -1:
                continue
            else:
                for a in actions:
            
                    if a == 'u':
                        if i == 0:
                            i_v = 0
                        else:
                            i_v = i - 1
                        j_v = j
                            
                    elif a == 'd':
                        if i == height - 1:
                            i_v = height - 1
                        else:
                            i_v = i + 1
                        j_v = j
                            
                    elif a == 'l':
                        i_v = i
                        if j == 0:
                            j_v = 0
                        else:
                            j_v = j - 1
                    
                    elif a == 'r':
                        i_v = i
                        if j == width - 1:
                            j_v = width - 1
                        else:
                            j_v = j + 1
                    
                    V[i, j] = V[i, j] + 1/4 * (REWARD + GAMMA * V_static[i_v, j_v])
    
    return V        
        

height = 4
width = 4

actions = (['u', 'd', 'l', 'r'])
#terminal_states = [(0,0), (3,3)]
V = np.zeros((height,width))

num_swipes = 0
while True:
    V_static = V.copy()    
    V = sweep_once(V, V_static, height, width, actions)
    num_swipes += 1
    if np.sum(np.abs(V - V_static)) < THRESHOLD:
        break

print("Total swipes = ", num_swipes, "\n")
print_value_table(V)


