#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
import slim.nets.inception_v1 as inception_v1
import scipy.io as scio
import csv
from collections import defaultdict
import json

from create_tf_record import *
import tensorflow.contrib.slim as slim

label0=0
label1=0
label2=0
label3=0
label4=0
label5=0
label6=0
label7=0
label8=0
label9=0
label10=0
label11=0
feature_vector = np.zeros(shape=(1024))
user_label0="user0"
user_label1="user1"
user_label2="user2"
user_label3="user3"
user_label4="user4"
user_label5="user5"
user_label6="user6"
user_label7="user7"
user_label8="user8"
user_label9="user9"
user_label10="user10"
user_label11="user11"
user_label12="user12"
user_label13="user13"

d = {}
d[user_label0] = []
d[user_label1] = []
d[user_label2] = []
d[user_label3] = []
d[user_label4] = []
d[user_label5] = []
d[user_label6] = []
d[user_label7] = []
d[user_label8] = []
d[user_label9] = []
d[user_label10] = []
d[user_label11] = []
d[user_label12] = []
d[user_label13] = []





def  predict(models_path,labels_filename,labels_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[1, resize_height, resize_width, depths], name='input')


    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        out, end_points = inception_v1.inception_v1(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)
        #print(out)


    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    #print(score)

    class_id = tf.argmax(score, 1)



    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)

    image_dir = image_dir14


    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    global label0
    global label1
    global label2
    global label3
    global label4
    global label5
    global label6
    global label7
    global label8
    global label9
    global label10
    global label11
    for image_path in images_list:
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
        max_score=pre_score[0,pre_label]


        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(pb_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")


        fc_tensor = sess.graph.get_tensor_by_name('InceptionV1/Logits/Dropout_0b/Identity:0')
        feature = sess.run(fc_tensor, feed_dict={input_images:im})
        for j in range (1024):
           feature_vector[j] = feature[0,0,0,j]
        list=feature_vector.tolist()
        x=image_path
        x1 = x.split('\\',1)[1]
        #d[user_label].append(x1)
        list1=[x1]
        list1= list1+list
        #print(feature_vector.shape)

        with open(image_csv, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list1)

        #scio.savemat(image_path+ '.mat', {"feature_vector": feature_vector})



        print ("{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,labels[pre_label], max_score))
        if (pre_label == 0 and max_score > 0.7):
            label0 += 1
        elif (pre_label == 1 and max_score > 0.7):
            label1 += 1
        elif (pre_label == 2 and max_score > 0.7):
            label2 += 1
        elif (pre_label == 3 and max_score > 0.7):
            label3 += 1
        elif (pre_label == 4 and max_score > 0.7):
            label4 += 1
        elif (pre_label == 5 and max_score > 0.7):
            label5 += 1
        elif (pre_label == 6 and max_score > 0.7):
            label6 += 1
        elif (pre_label == 7 and max_score > 0.7):
            label7 += 1
        elif (pre_label == 8 and max_score > 0.7):
            label8 += 1
        elif (pre_label == 9 and max_score > 0.7):
            label9 += 1
        elif (pre_label == 10 and max_score > 0.7):
            label10 += 1
        elif (pre_label == 11 and max_score > 0.7):
            label11 += 1
    sess.close()


if __name__ == '__main__':

    class_nums=12
    image_dir0='user/user0'
    image_dir1='user/user1'
    image_dir2='user/user2'
    image_dir3='user/user3'
    image_dir4='user/user4'
    image_dir5='user/user5'
    image_dir6='user/user6'
    image_dir7='user/user7'
    image_dir8='user/user8'
    image_dir9='user/user9'
    image_dir10='user/user10'
    image_dir11='user/user11'
    image_dir12='user/user12'
    image_dir13='user/user13'
    image_dir14='user_test/user11/test/pos'
    #user_labe11=["user0","user1","user2","user3","user4","user5","user6","user7","user8","user9","user10","user11","user12","user13"]
    #image_dir=['user/user0','user/user1','user/user2','user/user3','user/user4','user/user5','user/user6','user/user7','user/user8','user/user9','user/user10','user/user11','user/user12','user/user13']
    labels_filename='dataset/label.txt'
    image_csv='user_test/user11/test/pos.csv'
    feedback_json='user/user_image.json'
    models_path='classification_model/best_models_40000_0.8319.ckpt'
    pb_path = "classification_model/pb/frozen_model.pb"

    batch_size = 1  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]

    predict(models_path, labels_filename, class_nums, data_format)

    #with open(feedback_json, "a") as f:
     #   json.dump(d, f)
    #f.close

    number = label0 + label1 + label2 + label3 + label4 + label5 + label6 + label7 + label8 + label9 + label10 + label11
    print('animal:%d,percentage:%f' % (label0,label0/number))
    print('art:%d,percentage:%f' % (label1,label1/number))
    print('building:%d,percentage:%f' % (label2,label2/number))
    print('car:%d,percentage:%f' % (label3,label3/number))
    print('food:%d,percentage:%f' % (label4,label4/number))
    print('makeup:%d,,percentage:%f' % (label5,label5/number))
    print('mountain:%d,percentage:%f' % (label6,label6/number))
    print('night:%d,percentage:%f' % (label7,label7/number))
    print('painting:%d,percentage:%f' % (label8,label8/number))
    print('plant:%d,percentage:%f' % (label9,label9/number))
    print('sea:%d,percentage:%f' % (label10,label10/number))
    print('human:%d,percentage:%f' % (label11,label11/number))
    #with open(feedback_json, "a") as f:
     #   json.dump(d, f)
    #f.close


