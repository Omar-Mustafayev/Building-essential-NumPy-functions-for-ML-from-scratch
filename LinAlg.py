import numpy as np

print("Hello, world!")

def shape(array):
    """
    Equivalent to np.shape(x)
    returns (n,m) where all vectors considered as matrix 
    """
    num_rows = len(array)
    num_columns = len(array[0])
    return (num_rows, num_columns)

def add(array1, array2):
    """ Description:
    sum of array1 and array2
    In Linear Algebra addition is done element-wise
    Equivalent to np.add(x,y) 

    Args:
        array1 (list): matrix
        array2 (list): matrix

    Raises:
        ValueError: if shapes doesn't match. Due to only same shapes of matrices can be added

    Returns:
        array: matrix
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
    """ Description:
    subtraction of array1 and array2
    In Linear Algebra subtraction is done element-wise
    Equivalent to np.subtract(x,y) 

    Args:
        array1 (list): matrix
        array2 (list): matrix

    Raises:
        ValueError: if shapes doesn't match. Due to only same shapes of matrices can be added

    Returns:
        array: matrix
    """
    return add(array1, scalar_multiply(array2,-1))

def scalar_multiply(array,scalar=1):
    """Description:
    multiplies array by scalar
    In Linear Algebra scalar operates on every element in matrix
    Equivalent to np.multiply(x,scalar)

    Args:
        array (list): matrix
        scalar (int): number
    
    Returns:
        array: matrix
    """
    return [[elem * scalar for elem in row] for row in array]


def eye(shape_array):
    """Description:
    returns matrix of shape_array with 1s on the diagonal and 0s elsewhere
    In Linear Algebra this matrix is called identity matrix
    Equivalent to np.eye(n_row,n_col)

    Args:
        shape_array (tuple): shape of matrix

    Raises:
        ValueError: identity matrix shoyuld be square matrix, so n_row = n_col should be given

    Returns:
        array: identity matrix
    """
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
    """Description:
    multiplies array1 and array2
    In Linear Algebra multiplication is done by taking dot product of rows and columns. Formula is sum(a[i][k] * b[k][j]) for k from 0 to n_col1, i from 0 to n_row1, j from 0 to n_col2
    """
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
    """Description:
    returns matrix of shape_array with 0s
    In Linear Algebra this matrix is called zero matrix
    Equivalent to np.zeros((n_row,n_col))
    Args:
        shape_array (tuple): shape of matrix
    
    Returns:
        array: zero matrix
    """
    n_row,n_col = shape_array
    zero_matrix = []
    for i in range(n_row):
        row = []
        for j in range(n_col):
            row.append(0)
        zero_matrix.append(row)    
    
    return zero_matrix

def transpose(array):
    """Description:
    returns transposed matrix of given matrix
    In Linear Algebra transposed matrix is obtained by swapping rows and columns. Formula is a[i][j] = a[j][i]
    Equivalent to np.transpose(x)
    
    Args:
        array (list): matrix
    
    Returns:
        array: transposed matrix
    """
    n_row,n_col = shape(array)
    matrix = zeros((n_col,n_row))
    for i in range(n_row):
        for j in range(n_col):
            matrix[j][i] = array[i][j]
    return matrix

def diag(array, k=0):
    """Description:
    if array is 1D vector, returns diagonal matrix with given vector on the diagonal
    if array is 2D matrix, returns k-th diagonal of given matrix
    In Linear Algebra diagonal matrix is obtained by putting given vector on the diagonal and 0s elsewhere. Formula is a[i][j] = array[i] if i == j + k else 0
    Equivalent to np.diag(x,k)
    Args:
        array (list): matrix or vector
        k (int): k-th diagonal. Default is 0, which is main diagonal
    Returns:
        array: diagonal matrix or k-th diagonal of given matrix
    """
    try:
        n_row, n_col = shape(array)
        diagonal = []
        for i in range(n_row):
            for j in range(n_col):
                if i == (j - k) and (0 <= j- k <= n_col):
                    diagonal.append(array[i][j])
        return diagonal
    except TypeError:
        result_matrix = zeros((len(array),len(array)))
        for i in range(len(array)):
            result_matrix[i][i] = array[i]
        return result_matrix
        
    
def norm(array, type):
    """_summary_

    Args:
        array (list): 1D vector
        type (string): type of norm
        - "manhattan" or "l1" for Manhattan norm
        - "euclidean" or "l2" for Euclidean norm

    Raises:
        ValueError: if given array is not 1D vector
    
    Returns:
        int: corresponding norm of vector
    """
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
            
            

x = [[1],[2],[3]]
y = [[1,2,3]]
A = [[1,2,3],[4,5,6]]
A1 = [[3,2,1],[6,5,4]]
B = [[1,2],[3,4],[5,6]]
C = [[1,2],[2,4]]

print("Testing shape function:")
print("Custom:", shape(A), "Numpy:", np.shape(A))

print("\nTesting add function:")
print("Custom:", add(A, A1), "Numpy:", np.add(A, A1).tolist())

print("\nTesting subtract function:")
print("Custom:", subtract(A, A1), "Numpy:", np.subtract(A, A1).tolist())

print("\nTesting scalar_multiply function:")
print("Custom:", scalar_multiply(A, 2), "Numpy:", np.multiply(A, 2).tolist())

print("\nTesting eye function:")
print("Custom:", eye((3, 3)), "Numpy:", np.eye(3).tolist())

print("\nTesting mat_multiply function:")
print("Custom:", mat_multiply(A, B), "Numpy:", np.matmul(A, B).tolist())

print("\nTesting zeros function:")
print("Custom:", zeros((2, 3)), "Numpy:", np.zeros((2, 3)).tolist())

print("\nTesting transpose function:")
print("Custom:", transpose(A), "Numpy:", np.transpose(A).tolist())

print("\nTesting diag function:")
print("Custom:", diag([1, 2, 3]), "Numpy:", np.diag([1, 2, 3]).tolist())
print("Custom:", diag(C), "Numpy:", np.diag(C).tolist())

print("\nTesting norm function:")
print("Custom (Manhattan):", norm(x, "manhattan"), "Numpy:", np.linalg.norm(x, 1))
print("Custom (Euclidean):", norm(x, "euclidean"), "Numpy:", np.linalg.norm(x, 2))