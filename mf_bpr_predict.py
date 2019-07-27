import numpy as np
import csv
import tensorflow as tf

import heapq


i=7


if(i==0):
    path="user_test/user0/"

elif(i==1):
    path="user_test/user1/"

elif (i == 2):
    path = "user_test/user2/"

elif(i==3):
    path="user_test/user3/"

elif(i==4):
    path="user_test/user4/"

elif(i==5):
    path="user_test/user5/"

elif(i==6):
    path="user_test/user6/"

elif(i==7):
    path="user_test/user7/"

elif(i==8):
    path="user_test/user8/"

elif(i==9):
    path="user_test/user9/"

elif(i==10):
    path="user_test/user10/"

elif(i==11):
    path="user_test/user11/"




def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def vbpr_predict(itemFea_matrix, W1, b1, W2, b2):

    y1=np.matmul(itemFea_matrix,W1)+b1
    y2=np.matmul(y1,W2)+b2


    bpr_score=sigmoid(y2)


    return  y2,bpr_score



itemEmb_W1 = np.loadtxt(path+'model/itemEmb_W1.csv', delimiter=',')
itemEmb_b1 = np.loadtxt(path+'model/itemEmb_b1.csv', delimiter=',')
itemEmb_W2 = np.loadtxt(path+'model/itemEmb_W2.csv', delimiter=',')
itemEmb_b2 = np.loadtxt(path+'model/itemEmb_b2.csv', delimiter=',')


image_feature_path1=path+'test/pos.csv'
image_feature_path2=path+'test/neg.csv'

csv_reader1 = csv.reader(open(image_feature_path1))
csv_reader2 = csv.reader(open(image_feature_path2))


k=0
j=0
N=36
L=36
f=0


pos = np.zeros(L)
neg=np.zeros(L)
rec=np.zeros(2*L)



for item in csv_reader1:
    item_id = item[0]
    item_feature = item[1:]
    item_feature1 = item_feature
    item_feature1 = map(float, item_feature1)

    item_array = np.array(list(item_feature1))
    item_array = np.reshape(item_array, (1, 1024))

    W1=itemEmb_W1
    b1=itemEmb_b1
    W2 = itemEmb_W2
    b2=itemEmb_b2

    uij1,score1=vbpr_predict(item_array, W1, b1, W2, b2)
    pos[k]=uij1[0]
    rec[k]=uij1[0]
    k+=1


for item in csv_reader2:
    item_id = item[0]
    item_feature = item[1:]
    item_feature1 = item_feature
    item_feature1 = map(float, item_feature1)
    # for i in range(1024):
    #  item_feature1[i]=float(item_feature1[i])
    item_array = np.array(list(item_feature1))
    item_array = np.reshape(item_array, (1, 1024))

    W1 = itemEmb_W1
    b1 = itemEmb_b1
    W2 = itemEmb_W2
    b2 = itemEmb_b2

    uij2, score2 = vbpr_predict(item_array, W1, b1, W2, b2)
    neg[j]=uij2[0]
    rec[k]=uij2[0]
    j+=1
    k+=1


min = np.min(rec)

if(min<0):
    for x in range(L):
        pos[x] = pos[x]-min
        rec[x] = rec[x]-min

    for x in range(L):
        neg[x] = neg[x]-min
        rec[x+L] = rec[x+L]-min

print("pos_score:")
print(pos)


print("neg_score:")
print(neg)

#print(rec)

re1 = map(list(rec).index, heapq.nlargest(N, list(rec)))  # 求最大的三个索引    nsmallest与nlargest相反，求最小

re2 = heapq.nlargest(N, rec)  # 求最大的三个元素


print("topN_score:")
print(re2)
print("topN_index:")
rec1=list(re1)
print(rec1)



rec2=np.array(rec1)


for h in range(N):
    if rec2[h]<L:
        f+=1

print("precision:")
print(f/N)



