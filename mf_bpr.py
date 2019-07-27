# -*- coding:utf-8 -*-

import csv
import numpy as np
import random
import json
import tensorflow as tf

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
        print(self.l)
        for item in self.imageFeatures_pos:
            try:
                self.imageFeaMatrix_pos[self.item_dict_pos[item]] = self.imageFeatures_pos[item]
            except:
                pass
        print(self.imageFeaMatrix_pos)


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


def get_variable(type, shape, mean, stddev, name):
    if type == 'W':
        var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                              initializer=tf.random_normal_initializer(mean=mean, stddev=stddev))
        tf.add_to_collection('regular_losses', tf.contrib.layers.l2_regularizer(0.005)(var))
        return var
    elif type == 'b':
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                               initializer=tf.zeros_initializer())


def VBPR(itemFea_matrix_pos,itemFea_matrix_neg,pos_item_idx,neg_item_idx,L):

    itemEmb_W1 = get_variable(type='W', shape=[model.imageFeatureDim, model.k], mean=0, stddev=0.01, name='itemEmb_W1')
    itemEmb_b1 = get_variable(type='b', shape=[model.k], mean=0, stddev=0.01, name='itemEmb_b1')

    itemEmb_W2 = get_variable(type='W', shape=[ model.k ,1], mean=0, stddev=0.01, name='itemEmb_W2')
    itemEmb_b2 = get_variable(type='b', shape=[1], mean=0, stddev=0.01, name='itemEmb_b2')


    W1 = itemEmb_W1
    b1 = itemEmb_b1

    W2 = itemEmb_W2
    b2 = itemEmb_b2

    x1 = itemFea_matrix_pos
    x2 = tf.gather(itemFea_matrix_neg, neg_item_idx)

    y1_1 = tf.matmul(x1, W1) + b1
    y1_2 = tf.matmul(x2, W1) + b1

    y2_1 = tf.matmul(y1_1, W2) + b2
    y2_2 = tf.matmul(y1_2, W2) + b2

    BPR_loss = tf.reduce_mean(-tf.log(tf.sigmoid(y2_1-y2_2)))

    return BPR_loss, y2_1,y2_2



i=11
if(i==0):
    pos_path="user_test/user0/pos.csv"
    neg_path="user_test/user0/neg.csv"
    model_path="user_test/user0/"
elif(i==1):
    pos_path="user_test/user1/pos.csv"
    neg_path="user_test/user1/neg.csv"
    model_path="user_test/user1/"
elif (i == 2):
    pos_path = "user_test/user2/pos.csv"
    neg_path = "user_test/user2/neg.csv"
    model_path="user_test/user2/"
elif(i==3):
    pos_path="user_test/user3/pos.csv"
    neg_path="user_test/user3/neg.csv"
    model_path="user_test/user3/"
elif(i==4):
    pos_path="user_test/user4/pos.csv"
    neg_path="user_test/user4/neg.csv"
    model_path="user_test/user4/"
elif(i==5):
    pos_path="user_test/user5/pos.csv"
    neg_path="user_test/user5/neg.csv"
    model_path="user_test/user5/"
elif(i==6):
    pos_path="user_test/user6/pos.csv"
    neg_path="user_test/user6/neg.csv"
    model_path="user_test/user6/"
elif(i==7):
    pos_path="user_test/user7/pos.csv"
    neg_path="user_test/user7/neg.csv"
    model_path="user_test/user7/"
elif(i==8):
    pos_path="user_test/user8/pos.csv"
    neg_path="user_test/user8/neg.csv"
    model_path="user_test/user8/"
elif(i==9):
    pos_path="user_test/user9/pos.csv"
    neg_path="user_test/user9/neg.csv"
    model_path="user_test/user9/"
elif(i==10):
    pos_path="user_test/user10/pos.csv"
    neg_path="user_test/user10/neg.csv"
    model_path="user_test/user10/"
elif(i==11):
    pos_path="user_test/user11/pos.csv"
    neg_path="user_test/user11/neg.csv"
    model_path="user_test/user11/"


model=Processing(K=512)
#model.load_training_data('user/user_image.json')  ## feedback_file.json
model.load_image_feature_pos(pos_path)  ## image_feature.csv
#print(model.imageFeaMatrix_pos)
model.load_image_feature_neg(neg_path)
#print(model.imageFeaMatrix)

itemFea_matrix_pos = tf.placeholder(dtype=tf.float32, shape=[None, model.imageFeatureDim])
itemFea_matrix_neg = tf.placeholder(dtype=tf.float32, shape=[None, model.imageFeatureDim])

pos_item_idx = tf.placeholder(dtype=tf.int32, shape=[None,])
neg_item_idx = tf.placeholder(dtype=tf.int32, shape=[None,])
user_select = tf.placeholder(dtype=tf.int32, shape=[None,])
L=tf.placeholder(dtype=tf.int32, shape=[])

BPR_loss, ui, uj = VBPR(itemFea_matrix_pos,itemFea_matrix_neg,pos_item_idx,neg_item_idx,L )
tf.add_to_collection('BPR_losses', BPR_loss)
regular_loss = tf.add_n(tf.get_collection('regular_losses'))
loss = BPR_loss


global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(0.001, global_step, decay_steps=100, decay_rate=0.80, staircase=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=5)


loss_txt='user/user1_1/user1_loss.txt'
acc_txt='user/user1_1/user1_acc.txt'
min_loss=1
max_acc=0
for step in range(20001):


    step_now = step
    neg_item = []
    AUC=0
    acc=0
    l=model.l
    r=model.r
    #print(l)
    while len(neg_item)< l:
        a=random.randint(0, r-1)
        if a not in neg_item:
            neg_item.append(a)

    w1=[x for x in range(i*64,(i+1)*64)]
    #print(w1)
    _, los, B_loss, l2_loss, ui1,uj1= sess.run([train_step, loss, BPR_loss, regular_loss, ui, uj],
                          feed_dict={itemFea_matrix_pos: model.imageFeaMatrix_pos,
                                     itemFea_matrix_neg: model.imageFeaMatrix_neg,
                                     neg_item_idx: neg_item,
                                     L:l,
                                     global_step: step_now})

    for k in range(len(neg_item)):
        if ui1[k]-uj1[k]>10:
            AUC+=1
    AUC=AUC/l
    #print(un1)

    print('step-%d, loss : %f，accuracy：%f'%(step_now, los, AUC))

    if(AUC>=max_acc):
        print('saving model. accuracy = %f , loss = %f' % (AUC,los))
        saver.save(sess, model_path+'model/model_%f.ckpt' % (AUC), global_step=step)
        max_acc = AUC
        min_loss=los





"""
    pos_item = [x for x in range(model.nItems) if model.R[user_i][x] != 0]
    neg_item = []
    while len(neg_item)< 5*len(pos_item):
        a=random.randint(0, model.nItems-1)
        if a not in pos_item:
            neg_item.append(a)
"""



"""
                                     user_idx: user_i,
                                     pos_item_idx: pos_item,
                                     neg_item_idx: neg_item,
"""

"""    
with open(loss_txt, "a+") as  f1:
        f1.write(str(step_now))
        f1.write("\t")
        f1.write(str(los))
        f1.write("\n")
with open(acc_txt, "a+") as  f2:
        f2.write(str(step_now))
        f2.write("\t")
        f2.write(str(acc))
        f2.write("\n")
    #f2.close()
"""