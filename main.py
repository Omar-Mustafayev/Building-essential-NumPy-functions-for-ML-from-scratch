import numpy as np

print("Hello, world!")

x = [[1],[2],[3]]
y = [4,5,6]
A = [[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]

def shape(array):
    num_rows = len(array)
    try:
        num_columns = len(array[0])
    except:
        num_columns = None
    return num_rows, num_columns
    
print(shape(A))
print(np.shape(A))
print(shape(x))
print(np.shape(np.transpose(x)))
print(x)