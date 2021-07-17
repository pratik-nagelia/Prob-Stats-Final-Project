import numpy as np

def steady_state_power(transition_matrix):
        result = np.linalg.matrix_power(transition_matrix,150)
        print ("Steady State: Power Iteration >> ", result[0])
        return result

# Sample Transition matrix according to the harcopy answer submitted
a = np.array([[0.9,0.1,0,0],[0,0,0.5,0.5],[0,0,0.9,0.1],[0.8,0.2,0,0]])
print (steady_state_power(a))