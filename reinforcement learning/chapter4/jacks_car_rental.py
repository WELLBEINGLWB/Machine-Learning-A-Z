import numpy as np
#from scipy.stats import poisson
import math

MAX_CARS = 20
REQ_L1 = 3
REQ_L2 = 4
RET_L1 = 3
RET_L2 = 2
MOVE_CAR_COST = 2
MAX_MOVE_CARS = 5
GAMMA = 0.9
THRESHOLD = 1e-1
RENTAL_CREDIT = 10
POISSON_UP_BOUND = 10

actions = np.arange(-MAX_MOVE_CARS, MAX_MOVE_CARS+1)
states = []
for i in range(MAX_CARS+1):
    for j in range(MAX_CARS+1):
        states.append([i,j])

stateValue = np.zeros((MAX_CARS+1, MAX_CARS+1))

policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

poissonBackup = dict()
def poisson(n, lam):
    """ returns discrete poisson possibility using dict to save computation time """
    global poissonBackup
    key = n * 10 + lam
    if key not in poissonBackup.keys():
        poissonBackup[key] = math.exp(-lam) * (lam**n) / math.factorial(n)
    return poissonBackup[key]

def expectedReturn(state, action, stateValue):
    returns = 0.0

    returns -= abs(action) * MOVE_CAR_COST
    # go through all possible rental requests
    for rentalReqL1 in range(POISSON_UP_BOUND):
        for rentalReqL2 in range(POISSON_UP_BOUND):
            # moving cars form one location to the other
            numCarsL1 = int(min(state[0] - action, MAX_CARS))
            numCarsL2 = int(min(state[1] + action, MAX_CARS))
            # only available cars can be requested
            realRentalReqL1 = min(numCarsL1, rentalReqL1)
            realRentalReqL2 = min(numCarsL2, rentalReqL2)
            # calculating the immediate reward for requested cars
            reward = (realRentalReqL1 + realRentalReqL2) * RENTAL_CREDIT
            # recalculating available cars
            numCarsL1 -= realRentalReqL1
            numCarsL2 -= realRentalReqL2
            # calculating the probability that the combination of cars is requested
            # at location one and two
            prob = poisson(rentalReqL1, REQ_L1) * poisson(rentalReqL2, REQ_L2)
            # to save computation time constant returned cars can be calculated
            constantReturnedCars = False
            if constantReturnedCars == True:
                returnedCarsL1 = RET_L1
                returnedCarsL2 = RET_L2
                # make sure that maximum of 20 cars is available at each location
                numCarsL1 = min(numCarsL1 + returnedCarsL1, MAX_CARS)
                numCarsL2 = min(numCarsL2 + returnedCarsL2, MAX_CARS)
                # add the cumulative reward
                returns += prob * (reward + GAMMA * stateValue[numCarsL1, numCarsL2])
            else:
                # create static values for iterations
                numCarsL1Static = numCarsL1
                numCarsL2Static = numCarsL2
                probStatic = prob
                for returnedCarsL1 in range(POISSON_UP_BOUND):
                    for returnedCarsL2 in range(POISSON_UP_BOUND):
                        # reassign static values
                        numCarsL1 = numCarsL1Static
                        numCarsL2 = numCarsL2Static
                        prob = probStatic
                        # make sure that maximum of 20 cars at each location
                        numCarsL1 = min(numCarsL1Static + returnedCarsL1, MAX_CARS)
                        numCarsL2 = min(numCarsL2Static + returnedCarsL2, MAX_CARS)
                        # calculate the composite probability of requested and rented cars in each case
                        prob_ = poisson(returnedCarsL1, RET_L1) * \
                                poisson(returnedCarsL2, RET_L2) * \
                                prob
                        returns += prob_ * (reward + GAMMA * stateValue[numCarsL1, numCarsL2])

    return returns


newStateValue = np.zeros((MAX_CARS+1, MAX_CARS+1))
improvePolicy = False
policyImprovements = 0

while True:
    if improvePolicy == True:
        print("Policy improvement", policyImprovements)
        policyImprovements += 1
        newPolicy = np.zeros((MAX_CARS+1, MAX_CARS+1))
        for i, j in states:
            actionReturns = []

            for action in actions:
                if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                    actionReturns.append(expectedReturn([i,j], action, stateValue))
                else:
                    actionReturns.append(-float('inf'))
            bestAction = np.argmax(actionReturns)
            newPolicy[i, j] = actions[bestAction]

        policyChanges = np.sum(newPolicy != policy)
        print("Policy changed for ", policyChanges, "states")
        if policyChanges == 0:
            policy = newPolicy
            break
        policy = newPolicy
        improvePolicy = False

    # start policy evaluation
    for i, j in states:
        newStateValue[i, j] = expectedReturn([i, j], policy[i, j], stateValue)
    if np.sum(np.abs(newStateValue - stateValue)) < THRESHOLD:
        stateValue[:] = newStateValue
        improvePolicy = True
        continue
    stateValue[:] = newStateValue