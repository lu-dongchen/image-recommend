
import numpy as np
import json
import csv
import heapq

i=4



if(i==0):
    pos_path="user_test/user0/"
    neg_path="user_test/user0/"
elif(i==1):
    pos_path="user_test/user1/"
    neg_path="user_test/user1/"
elif (i == 2):
    pos_path = "user_test/user2/"
    neg_path = "user_test/user2/"
elif(i==3):
    pos_path="user_test/user3/"
    neg_path="user_test/user3/"
elif(i==4):
    pos_path="user_test/user4/"
    neg_path="user_test/user4/"
elif(i==5):
    pos_path="user_test/user5/"
    neg_path="user_test/user5/"
elif(i==6):
    pos_path="user_test/user6/"
    neg_path="user_test/user6/"
elif(i==7):
    pos_path="user_test/user7/"
    neg_path="user_test/user7/"
elif(i==8):
    pos_path="user_test/user8/"
    neg_path="user_test/user8/"
elif(i==9):
    pos_path="user_test/user9/"
    neg_path="user_test/user9/"
elif(i==10):
    pos_path="user_test/user10/"
    neg_path="user_test/user10/"
elif(i==11):
    pos_path="user_test/user11/"
    neg_path="user_test/user11/"

class Processing:
    def __init__(self, K):
        self.imageFeatures_pos = {}
        self.item_dict_pos = {}
        self.imageFeatures_neg = {}
        self.item_dict_neg = {}
        self.imageFeaMatrix_pos = []
        self.imageFeaMatrix_neg = []
        self.imageFeatureDim = 1024
        self.k = K   # Latent dimension
        self.l=0
        self.r=0

    # def load_data(self, image_feature_path, rating_file_path):
    #     # self.load_image_feature(image_feature_path)
    #     self.load_training_data(rating_file_path)

    def load_image_feature_pos(self, image_feature_path):
        csv_reader=csv.reader(open(image_feature_path))
        for item in csv_reader:
            item_id=item[0]
            item_feature = item[1:]
            item_feature = list(map(float, item_feature))
            self.imageFeatures_pos[item_id] = item_feature
            if item_id not in self.item_dict_pos.keys():
                self.item_dict_pos[item_id] = self.l
            self.l+=1

        self.imageFeaMatrix_pos=[[0.]*self.imageFeatureDim]*self.l
        #print(self.l)
        for item in self.imageFeatures_pos:
            try:
                self.imageFeaMatrix_pos[self.item_dict_pos[item]] = self.imageFeatures_pos[item]
            except:
                pass
        #print(self.imageFeaMatrix_pos)


    def load_image_feature_neg(self, image_feature_path):
        csv_reader=csv.reader(open(image_feature_path))
        for item in csv_reader:
            item_id=item[0]
            item_feature = item[1:]
            item_feature = list(map(float, item_feature))
            self.imageFeatures_neg[item_id] = item_feature
            if item_id not in self.item_dict_neg.keys():
                self.item_dict_neg[item_id] = self.r
            self.r+=1

        self.imageFeaMatrix_neg=[[0.]*self.imageFeatureDim]*self.r
        for item in self.imageFeatures_neg:
            try:
                self.imageFeaMatrix_neg[self.item_dict_neg[item]] = self.imageFeatures_neg[item]
            except:
                pass
        #print(self.imageFeaMatrix_neg)

i=0
j=0
k=0
f=0
N =20
L=36


model=Processing(K=512)
model.load_image_feature_pos(pos_path+"pos.csv")  ## image_feature.csv
#print(model.imageFeaMatrix_pos)
model.load_image_feature_neg(neg_path+"neg.csv")



a=model.imageFeaMatrix_pos

#print(model.imageFeaMatrix)
b=np.mean(a, axis=0)

print(b)





pos = np.zeros(L)
neg=np.zeros(L)
rec=np.zeros(2*L)


image_feature_path1=pos_path+'test/pos.csv'
image_feature_path2=neg_path+'test/neg.csv'



csv_reader1 = csv.reader(open(image_feature_path1))
csv_reader2 = csv.reader(open(image_feature_path2))

for item in csv_reader1:
    num = 0
    denom = 0
    item_id = item[0]
    item_feature = item[1:]
    item_feature1 = item_feature
    item_feature1 = map(float, item_feature1)
    item_array = np.array(list(item_feature1))
    item_array = np.reshape(item_array, (1, 1024))
    num = float(np.sum(b * item_array))
    denom = np.linalg.norm(b) * np.linalg.norm(item_array)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    pos[j]=sim
    rec[j]=sim
    j+=1



for item in csv_reader2:
    num = 0
    denom = 0
    item_id = item[0]
    item_feature = item[1:]
    item_feature1 = item_feature
    item_feature1 = map(float, item_feature1)
    item_array = np.array(list(item_feature1))
    item_array = np.reshape(item_array, (1, 1024))
    num = float(np.sum(b * item_array))
    denom = np.linalg.norm(b) * np.linalg.norm(item_array)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    neg[k]=sim
    rec[j]=sim
    k+=1
    j+=1



re1 = map(list(rec).index, heapq.nlargest(N, list(rec)))  # 求最大的三个索引    nsmallest与nlargest相反，求最小

re2 = heapq.nlargest(N, rec)  # 求最大的三个元素

print("pos_score:")
print(pos)

print("neg_score:")
print(neg)



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