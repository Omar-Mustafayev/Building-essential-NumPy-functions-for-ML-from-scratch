import numpy as np

print("Hello, world!")

y1 = [[1,2,3]]
x = [[1],[2],[3]]
y = [[4,5,6]]
A = [[1,2,3],[4,5,6]]
A1 = [[3,2,1],[6,5,4]]
B = [[1,2],[3,4],[5,6]]
C = [[1,2],[2,4]]

def shape(array):
    """
    np.shape(x)
    returns (n,m) where all vectors considered as matrix 
    """
    num_rows = len(array)
    num_columns = len(array[0])
    return (num_rows, num_columns)

def add(array1, array2):
    """ Description:
    sum of array1 and array2
    Equivalent to np.add(x,y) 

    Args:
        array1 (list): matrix
        array2 (list): matrix

    Raises:
        ValueError: if shapes doesn't match. Due to only same shapes of matrices can be added

    Returns:
        array: result
    """
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
    return add(array1, scalar_multiply(array2,-1))

def scalar_multiply(array,scalar=1):
    return [[elem * scalar for elem in row] for row in array]
  
  
def norm(array, type):
    if type == "manhattan" or type == "l1":
        return sum([elem for elem in row] for row in array)
    elif type == "euclidean" or type == "l2":
        return sum([(elem**2)**0.5 for elem in row] for row in array)
#print(norm(y,"manhattan"))
def eye(shape_array):
    n_row,n_col = shape_array

    if n_row != n_col:
        raise ValueError("Identity matrices are square matrices")
    identity_matrix = []
    for i in range(n_row):
        row = []
        for j in range(n_col):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        identity_matrix.append(row)
    return identity_matrix
true_eye = eye(shape_array=(3,3))
#false_eye = eye(shape(x))
print(true_eye)
print(shape(true_eye))
true_eye = eye(shape_array=shape(C))
#false_eye = eye(shape(x))
print(true_eye)
print(shape(true_eye))