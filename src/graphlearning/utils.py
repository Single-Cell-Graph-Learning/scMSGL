import numpy as np

from numba import njit
from scipy.sparse import csr_matrix

@njit
def _rowsum_mat_entries(n: int):
    M = int(n*(n-1)) # number of non-diagonel entries in the matrix
    rows = np.zeros((M, ), dtype=np.int64)
    cols = np.zeros((M, ), dtype=np.int64)
    
    offset_1 = 0 # column offset for block of ones in rows
    offset_2 = 0 # column offset for individual ones in rows
    indx = 0
    
    for row in range(n):
        rows[indx:(indx+n-row-1)] = row
        cols[indx:(indx+n-row-1)] = offset_1 + np.arange(n-row-1)
        
        indx += n-row-1
        offset_1 += n-row-1
        
        if row>0:
            rows[indx:(indx+n-row)] = np.arange(row, n)
            cols[indx:(indx+n-row)] = offset_2 + np.arange(n-row)
            
            indx += n-row
            offset_2 += n-row
    return rows, cols

def rowsum_mat(n: int):   
    rows, cols = _rowsum_mat_entries(n)
    M = len(rows)
    return csr_matrix((np.ones((M, )), (rows, cols)), shape=(n, int(M/2))) 