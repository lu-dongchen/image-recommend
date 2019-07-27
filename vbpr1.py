# -*- coding:utf-8 -*-

import csv
import numpy as np
import random
import json
import tensorflow as tf

class Processing:
    def __init__(self, K, K2):
        self.R = []  # Rating matrix
        self.nUsers = 0
        self.nItems = 0
        self.user_dict = {}
        self.item_dict = {}
        self.imageFeatures = {}
        self.imageFeaMatrix = []
        self.imageFeatureDim = 1024
        self.k = K   # Latent dimension
        self.k2 = K2  # Visual dimension
        self.MF_loss = 0

    # def load_data(self, image_feature_path, rating_file_path):
    #     # self.load_image_feature(image_feature_path)
    #     self.load_training_data(rating_file_path)

    def load_image_feature(self, image_feature_path):
        csv_reader=csv.reader(open(image_feature_path))
        for item in csv_reader:
            item_id=item[0]
            item_feature = item[1:]
            item_feature = list(map(float, item_feature))
            self.imageFeatures[item_id] = item_feature
        self.imageFeaMatrix=[[0.]*self.imageFeatureDim]*self.nItems
        for item in self.imageFeatures:
            try:
                self.imageFeaMatrix[self.item_dict[item]] = self.imageFeatures[item]
            except:
                pass

    def load_training_data(self, rating_file_path):
        with open(rating_file_path,'r') as f:
            data = json.load(f)

        # create user/item idx dictionary
        for user_id in data:
            if user_id not in self.user_dict.keys():
                self.user_dict[user_id] = self.nUsers
                self.nUsers += 1
                for item_id in data[user_id]:
                    if item_id not in self.item_dict.keys():
                        self.item_dict[item_id] = self.nItems
                        self.nItems += 1
        self.R = np.array([[0.] * self.nItems] * self.nUsers)
        for user_id in data:
            for item_id in data[user_id]:
                self.R[self.user_dict[user_id], self.item_dict[item_id]] = 10 #data[user_id][item_id][1]

def get_variable(type, shape, mean, stddev, name):
    if type == 'W':
        var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                              initializer=tf.random_normal_initializer(mean=mean, stddev=stddev))
        tf.add_to_collection('regular_losses', tf.contrib.layers.l2_regularizer(0.005)(var))
        return var
    elif type == 'b':
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                               initializer=tf.zeros_initializer())


def VBPR(itemFea_matrix,user_idx,pos_item_idx,neg_item_idx,N,L):

    itemEmb_W1 = get_variable(type='W', shape=[model.imageFeatureDim, model.k], mean=0, stddev=0.01, name='itemEmb_W1')
    itemEmb_b1 = get_variable(type='b', shape=[model.k], mean=0, stddev=0.01, name='itemEmb_b1')

    itemEmb_W2 = get_variable(type='W', shape=[ model.k ,1], mean=0, stddev=0.01, name='itemEmb_W2')
    itemEmb_b2 = get_variable(type='b', shape=[1], mean=0, stddev=0.01, name='itemEmb_b2')

    visual_I =itemFea_matrix
    #print(visual_I)

    W1 = itemEmb_W1
    b1 = itemEmb_b1

    W2 = itemEmb_W2
    b2 = itemEmb_b2

    x1 = tf.gather(visual_I, pos_item_idx)
    #print(x1)
    x2 = tf.gather(visual_I, neg_item_idx)

    y1_1 = tf.matmul(x1, W1) + b1
    y1_2 = tf.matmul(x2, W1) + b1

    y2_1 = tf.matmul(y1_1, W2) + b2
    y2_2 = tf.matmul(y1_2, W2) + b2

    ui=tf.concat([y2_1,y2_2],0)
    print(ui)
    uj=tf.reshape(ui,[1,2*L])
    print(uj)
    uk=tf.nn.top_k(uj,N)
    un=uk[1]
    BPR_loss = tf.reduce_mean(-tf.log(tf.sigmoid(y2_1-y2_2)))

    return BPR_loss, y2_1,y2_2,un





model=Processing(K=512, K2=32)
model.load_training_data('user/user_image.json')  ## feedback_file.json
model.load_image_feature('user/image_feature1024.csv')  ## image_feature.csv
#print(model.imageFeaMatrix)

itemFea_matrix = tf.placeholder(dtype=tf.float32, shape=[model.nItems, model.imageFeatureDim])
userFea_matrix = tf.placeholder(dtype=tf.float32, shape=[None, model.nItems])
user_idx = tf.placeholder(dtype=tf.int32, shape=[])
pos_item_idx = tf.placeholder(dtype=tf.int32, shape=[None,])
neg_item_idx = tf.placeholder(dtype=tf.int32, shape=[None,])
user_select = tf.placeholder(dtype=tf.int32, shape=[None,])
N=tf.placeholder(dtype=tf.int32, shape=[])
L=tf.placeholder(dtype=tf.int32, shape=[])

BPR_loss, ui, uj,un = VBPR(itemFea_matrix,user_idx,pos_item_idx,neg_item_idx, N,L )
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


    i=2

    step_now = step
    pos_item = [x for x in range(model.nItems) if model.R[i][x] != 0]
    neg_item = []
    AUC=0
    acc=0
    n=len(pos_item)
    l=len(pos_item)
    #print(l)
    while len(neg_item)< len(pos_item):
        a=random.randint(0, model.nItems-1)
        if a not in pos_item:
            if a not in neg_item:
                neg_item.append(a)

    w1=[x for x in range(i*64,(i+1)*64)]
    #print(w1)
    _, los, B_loss, l2_loss, ui1,uj1,un1= sess.run([train_step, loss, BPR_loss, regular_loss, ui, uj, un],
                          feed_dict={itemFea_matrix: model.imageFeaMatrix,
                                     userFea_matrix: model.R,
                                     user_idx: i,
                                     pos_item_idx: pos_item,
                                     neg_item_idx: neg_item,
                                     N:n,
                                     L:l,
                                     global_step: step_now})

    for k in range(len(neg_item)):
        if ui1[k]-uj1[k]>10:
            AUC+=1
    AUC=AUC/l
    #print(un1)
    uk1=np.reshape(un1,[l])
    for m in range (l):
        if (uk1[k]<l):
            acc+=1
    acc=acc/len(neg_item)
    print('step-%d, loss : %f，accuracy：%f'%(step_now, los,AUC))
    #print(AUC)

    if((AUC>=max_acc) and (los<min_loss)):
        print('saving model. accuracy = %f , loss = %f' % (AUC,los))
        saver.save(sess, './user/user1_1/model_%f.ckpt' % (AUC), global_step=step)
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