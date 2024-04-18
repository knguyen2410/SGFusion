import numpy as np
# array = [0.957, 0.903,0.859,0.792]
array = np.array([0.149,0.247,0.272,0.327])

def sigmoid(x):
    return 1/(1 + np.exp(-x))


min = min(array)
max = max(array)
for val in array:
    # a = ((val - min) / (max-min))
    a = val / np.sum(array)
    print(1-a)
