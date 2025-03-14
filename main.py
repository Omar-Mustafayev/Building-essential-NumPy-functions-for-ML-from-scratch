import numpy as np

print("Hello, world!")

y1 = [[1,2,3]]
x = [[1],[2],[3]]
y = [[4,5,6]]
A = [[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]

def shape(array):
    """
    returns (n,m) where all vectors considered as matrix 
    """
    num_rows = len(array)
    num_columns = len(array[0])
    return num_rows, num_columns


def sum(array1, array2):
    n_row, n_col = shape(array1)
    if (n_row,n_col) != shape(array2):
        raise ValueError("The two arrays should have the same shape")
    result = []
    for i in range(n_row):
        row = []
        for j in range(n_col):
            row.append(array1[i][j] + array2[i][j])
        result.append(row)
    return result

def subtract(array1, array2):
    n_row, n_col = shape(array1)
    if (n_row,n_col) != shape(array2):
        raise ValueError("The two arrays should have the same shape")
    result = []
    for i in range(n_row):
        row = []
        for j in range(n_col):
            row.append(array1[i][j] - array2[i][j])
        result.append(row)
    return result

c = sum(y1,y)
print(c)

c = subtract(y1,y)
print(c)