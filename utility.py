import numpy as np

def log_encode(x):
    if x > 0:
        return np.log(x+1)
    elif x < 0:
        return -np.log(-x-1)
    else:
        return 0