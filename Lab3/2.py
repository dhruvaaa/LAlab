import numpy as np
x=np.array([2,5,7])
y=np.array([1,1,-1])
A=np.zeros((3,2))
for i in range(3):
  A[i,0]=1
  A[i,1]=x[i]
A_pinv=np.linalg.pinv(A)
parameters= A_pinv @ y

a0 = parameters[0]
a1 = parameters[1]

print("a0 =", a0)
print("a1 =", a1)

# Decision boundary: a0 + a1*x = 0
x_boundary = -a0 / a1
print("Decision boundary =", x_boundary)

# Test image
X_test = 6

f = a0 + a1 * X_test

if f > 0:
    print("Test image belongs to Class 1")
else:
    print("Test image belongs to Class 2")
