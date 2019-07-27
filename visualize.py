
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

filename = 'user/user1/user1_acc.txt'
filename1 = 'user/user1/user1_loss.txt'
X ,Y = [] ,[]
X2 ,Y2 = [] ,[]

with open(filename, 'r') as f  :  # 1

    lines = f.readlines(  )  # 2

    for line in lines  :  # 3

        value = [float(s) for s in line.split()  ]  # 4

        X.append(value[0]  )  # 5

        Y.append(value[1])

with open(filename1, 'r') as f1  :  # 1

    lines = f1.readlines(  )  # 2

    for line in lines  :  # 3

        value = [float(s) for s in line.split()  ]  # 4

        X2.append(value[0]  )  # 5

        Y2.append(value[1])

print(X)

print(Y)
X1 = np.linspace(0,3000,3000)
Y1 = spline(X,Y,X1)

X3 = np.linspace(0,3000,3000)
Y3 = spline(X2,Y2,X3)

#plt.plot(X1, Y1)
plt.plot(X1, Y1,'r')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('loss')

plt.show()