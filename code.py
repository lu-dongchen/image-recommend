def VBPR(itemFea_matrix,user_idx,pos_item_idx,neg_item_idx,N,L):
    '''
    循环迭代训练过程
    :itemFea_matrix：图片特征向量
    :user_idx:       用户标签
    :pos_item_idx:   训练角户正样本
    :neg_i t em_i dx:训练用户负样本
    :N:              topN推荐个数，用于计算准确率
    :L:              正/负样本个数
    :return:         bpr_loss,topN_1(N个预测评分位置)
    '''
    #定义第一个隐含层
    itemEmb_W1 = get_variable(type='W', shape=[model.imageFeatureDim, model.k], mean=0, stddev=0.01, name='itemEmb_W1') #[1024,k]
    itemEmb_b1 = get_variable(type='b', shape=[model.k], mean=0, stddev=0.01, name='itemEmb_b1') #[k]
    #定义第二个隐含层
    itemEmb_W2 = get_variable(type='W', shape=[ model.k ,model.nUsers], mean=0, stddev=0.01, name='itemEmb_W2') #[k,12]
    itemEmb_b2 = get_variable(type='b', shape=[model.nUsers], mean=0, stddev=0.01, name='itemEmb_b2') #[12]

    visual_I =itemFea_matrix #[image_numbers,1024]
    #print(visual_I)

    W1 = itemEmb_W1 #[1024,k]
    b1 = itemEmb_b1 #[k]

    W2 = tf.gather (itemEmb_W2, user_idx) #[k, 1]
    b2 = tf.gather (itemEmb_b2, user_idx)  #[l]

    x1 = tf.gather(visual_I, pos_item_idx) #[?,1024],索引用户u正样本
    x2 = tf.gather(visual_I, neg_item_idx) #[?,1024],索引用户u负样本

    y1_1 = tf.matmul(x1, W1) + b1 #[?,k]
    y1_2 = tf.matmul(x2, W1) + b1 #[?,k]

    xui = tf.matmul(y1_1, W2) + b2 #[?,1],正样本预测分数
    xuj = tf.matmul(y1_2, W2) + b2 #[?,1],负样本预测分数

    xu=tf.concat([xui,xuj],0) #将正样本预测分数与负样本预测分数拼接为一个矩阵
    xu_1=tf.reshape(ui,[1,2*L]) #调整矩阵大小为[1,2*L]

    topN=tf.nn.top_k(xu_1,N) #索引最高N个得分，得到预测分数topN[0]及所在位置topN[1]
    topN_1=topN[1] #返回最高得分位置
    BPR_loss = tf.reduce_mean(-tf.log(tf.sigmoid(y2_1-y2_2))) #loss=1/E(U)  ∑_((i,j)∈E(U))[-logσ(x ̂(u,i)-x ̂(u,j) ) ]

    return BPR_loss,topN_1