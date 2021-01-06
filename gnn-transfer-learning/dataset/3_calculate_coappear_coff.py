import json
from tqdm import tqdm
import numpy as np
import heapq

matrix = np.loadtxt("dblp_v12_co_appear_matrix.txt")
coff_matrix = np.zeros((50, 50), dtype=np.float)
for row in range(50):
    for col in range(50):
        coff_matrix[row][col] = matrix[row][col] ** 2 / matrix[row][row] / matrix[col][col]

matrix = matrix.sum(axis=1)
result = heapq.nsmallest(6, range(len(matrix)), matrix.__getitem__)
print(result)
print(matrix[result])
