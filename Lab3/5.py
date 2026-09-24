import numpy as np

# Function to add a column of 1s
def add_ones(X):
    ones = np.ones((X.shape[0], 1))
    X_new = np.hstack((ones, X))
    return X_new


# Function to classify one input
def classify(X, W):
    X = np.hstack(([1], X))
    
    y = np.dot(W, X)
    
    if y > 0:
        return 1
    else:
        return 2


# Example data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 2]
])

Y = np.array([1, 1, -1, -1])


# Convert X into homogeneous coordinates
A = add_ones(X)

print("A =")
print(A)


# Find W using least squares
W = np.linalg.lstsq(A, Y, rcond=None)[0]

print("W =")
print(W)


# Classify the data
for i in range(len(X)):
    result = classify(X[i], W)
    print("X =", X[i], "Class =", result)
