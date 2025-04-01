import numpy as np

print("Hello, world!")


y = [[4,5,6]]


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

    Args:vectory
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
    n_row, n_col = shape(array)
    result = 0
    if n_col == 1:
        if type == "manhattan" or type == "l1": 
            for i in array:
                result += abs(i[0])
        elif type == "euclidean" or type == "l2":
            for i in array:
                result += (i[0]**2)
            result = result**0.5
    elif n_row == 1:
        return norm(transpose(array),type)
    else:
        raise ValueError("It is norm of vector!")
    return result
            
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


#def dot(array1, array2):
#    n_row1, n_col1 = shape(array1)
#   n_row2, n_col2 = shape(array2)
#  if n_col1 != n_row2:
#        raise ValueError("Neighbouring dimensions should match!")
#    sum = 0
#    for x in range(n_col1):
#        sum += array1[0][x] * array2[x][0]
#    return sum

def mat_multiply(array1, array2):
    n_row1, n_col1 = shape(array1)
    n_row2, n_col2 = shape(array2)
    if n_col1 != n_row2:
        raise ValueError("Neighbouring dimensions should match!")
    matrix = zeros((n_row1,n_col2))
    for i in range(n_row1):
        for j in range(n_col2):
            for k in range(n_col1): 
                matrix[i][j] += array1[i][k] * array2[k][j]
    return matrix

def zeros(shape_array):
    n_row,n_col = shape_array
    zero_matrix = []
    for i in range(n_row):
        row = []
        for j in range(n_col):
            row.append(0)
        zero_matrix.append(row)    
    
    return zero_matrix
    
y1 = [[1,2,3]]
x = [[1],[2],[3]]
A = [[1,2,3],[4,5,6]]
A1 = [[3,2,1],[6,5,4]]
B = [[1,2],[3,4],[5,6]]
C = [[1,2],[2,4]]



def transpose(array):
    n_row,n_col = shape(array)
    matrix = zeros((n_col,n_row))
    for i in range(n_row):
        for j in range(n_col):
            matrix[j][i] = array[i][j]
    return matrix
print(norm(x,"manhattan"))
print(norm(y1,"manhattan"))
print(norm(x,"euclidean"))
print(norm(y1,"euclidean"))
