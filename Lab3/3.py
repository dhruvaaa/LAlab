import numpy as np

x = [1, 2, 3]
y = [1, -1, 1]

# a0 + a1*x + a2*x^2 = y

A = np.zeros((3, 3))
b = np.zeros(3)

for i in range(3):
    A[i, 0] = 1
    A[i, 1] = x[i]
    A[i, 2] = x[i] * x[i]
    b[i] = y[i]

print("Initial matrix:")
print(A)
print(b)

# Gaussian elimination

factor = A[1, 0] / A[0, 0]

for j in range(3):
    A[1, j] = A[1, j] - factor * A[0, j]
b[1] = b[1] - factor * b[0]

factor = A[2, 0] / A[0, 0]

for j in range(3):
    A[2, j] = A[2, j] - factor * A[0, j]
b[2] = b[2] - factor * b[0]

factor = A[2, 1] / A[1, 1]

for j in range(3):
    A[2, j] = A[2, j] - factor * A[1, j]
b[2] = b[2] - factor * b[1]

print("Upper triangular matrix:")
print(A)
print(b)

# Back substitution

a2 = b[2] / A[2, 2]

a1 = (b[1] - A[1, 2] * a2) / A[1, 1]

a0 = (b[0] - A[0, 1] * a1 - A[0, 2] * a2) / A[0, 0]

print("a0 =", a0)
print("a1 =", a1)
print("a2 =", a2)

# Verify the three points

for i in range(3):
    value = a0 + a1 * x[i] + a2 * x[i] * x[i]
    print("For x =", x[i], "f(x) =", value, "actual y =", y[i])
