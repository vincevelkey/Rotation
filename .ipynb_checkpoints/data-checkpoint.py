import numpy as np

def generate_data(paradigm):
    '''
    Generate dataset as in Courville et. al. 2003. The parameter 'paradigm' can take
    the following values:
    - 'no-x'
    - 'few-x'
    - 'many-x'
    '''
    if paradigm == 'no-x':
        D = np.zeros((104, 4))
        D[:96,0] = 1
        D[:96,3] = 1
        D[96:,2] = 1
        D[96:,3] = 1

        return D
    elif paradigm == 'few-x':
        D = np.zeros((108, 4))
        D[:96, 0] = 1
        D[:96, 3] = 1
        D[96:100, 0] = 1
        D[96:100, 1] = 1
        D[100:, 2] = 1
        D[100:, 3] = 1
    elif paradigm == 'many-x':
        D = np.zeros((152, 4))
        D[:96, 0] = 1
        D[:96, 3] = 1
        D[96:144, 0] = 1
        D[96:144, 1] = 1
        D[144:, 2] = 1
        D[144:, 3] = 1
    else:
        raise ValueError("Unknown paradigm")