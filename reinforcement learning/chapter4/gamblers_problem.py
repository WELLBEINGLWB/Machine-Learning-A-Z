import numpy as np

PROB_HEADS = 0.4
CAPITAL_MAX = 100
DISCOUNT = 1 # undiscounted case
THRESHOLD = 1e-2
states = np.arange(CAPITAL_MAX + 1)

policy = np.zeros(CAPITAL_MAX + 1, dtype=int)

valueEstimate = np.zeros(CAPITAL_MAX + 1)
valueEstimate[CAPITAL_MAX] = 1

bestAction = np.zeros(CAPITAL_MAX + 1)

def valueIteration(states, valueEstimate):

    valueEstimateCopy = valueEstimate.copy()
    while True:

        delta = 0.0

        for s in states[1:CAPITAL_MAX]:
            actions = np.arange(0, min(s, CAPITAL_MAX - s) + 1)
            bestValue = 0
#            bestAction = np.zeros(CAPITAL_MAX + 1)
            actionReturns = []
            for a in actions:
                s_tails = s - a
                s_heads = s + a

                actionReturns.append(PROB_HEADS * valueEstimate[s_heads] + (1-PROB_HEADS) * valueEstimate[s_tails])
#
#                value = PROB_HEADS * valueEstimateCopy[s_heads] + \
#                        (1-PROB_HEADS) * valueEstimateCopy[s_tails]
#
#                if value > bestValue:
#                    bestValue = value
#                    bestAction[s] = a
#                    valueEstimate[s] = value
            newValue = np.max(actionReturns)
            valueEstimate[s] = newValue
            delta += np.abs(valueEstimate[s] - newValue)

#            valueEstimateCopy = valueEstimate.copy()

#        if np.sum(np.abs(valueEstimate - valueEstimateCopy)) < THRESHOLD:
        if delta < 1e-4:
            break

    # policy update
#    for s in states[1:CAPITAL_MAX]:
#        policy[s] = bestAction[s]


    return valueEstimate

valueEstimate = valueIteration(states, valueEstimate)

# policy update
for s in states[1:CAPITAL_MAX]:
    actions = np.arange(0, min(s, CAPITAL_MAX-s) + 1)
    actionReturns = []
    for a in actions:
        actionReturns.append(PROB_HEADS * valueEstimate[s + a] + (1-PROB_HEADS) * valueEstimate[s - a])
    policy[s] = actions[np.argmax(actionReturns)]




