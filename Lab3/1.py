import numpy as np

x=np.array([1,2,3])
y=np.array([1,2,2])
A=np.zeros((3,2))

for i in range(3):
  A[i,0]=1
  A[i,1]=x[i]
AT=A.T
ATA=np.dot(AT,A)
ATY=np.dot(AT,y)
W=np.dot(np.linalg.inv(ATA),ATY)
print(W)
print(np.dot(A,W))
print(y)
error = np.linalg.norm(np.dot(A, W) - y) ** 2

print("Total squared error =", error)
