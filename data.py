import numpy as np

def generate_data_second_order(axnum=1):
    '''
    Generate data for second-order conditioning task. The variable axnum gives the number of X-A pairings.
    '''

    D = np.zeros((210 + axnum, 4))
    D[:10, :] = [0,0,0,0]
    D[10:110, :] = [1,0,0,1]
    D[110:210, :] = [0, 0, 1,1]
    D[210:, :] = [1, 1, 0, 0]  # Additional rows for axnum > 0

    return D
