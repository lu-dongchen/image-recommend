import os
import csv
import numpy as np
from tensorflow.python import pywrap_tensorflow

path="user_test/user11/model/"
checkpoint_path = os.path.join(path+"model_0.766467.ckpt-3175")

# Read data from checkpoint file

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)

var_to_shape_map = reader.get_variable_to_shape_map()

# Print tensor name and values


for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(key.shape)
    if(key=='itemEmb_W1'):
        key1=reader.get_tensor(key)
        print(reader.get_tensor(key))
        key2=key1.reshape(1024,512)
        key3=key2.tolist()
        print(key3)
        key4=np.array(key3)

        np.savetxt(path+'itemEmb_W1.csv', key4 ,fmt='%.12f', delimiter=',')

for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(key.shape)
    if(key=='itemEmb_b1'):
        key1=reader.get_tensor(key)
        print(reader.get_tensor(key))
        key2=key1.reshape(1,512)
        key3=key2.tolist()
        print(key3)
        key4=np.array(key3)

        np.savetxt(path+'itemEmb_b1.csv', key4 ,fmt='%.12f', delimiter=',')



for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(key.shape)
    if(key=='itemEmb_W2'):
        key1=reader.get_tensor(key)
        print(reader.get_tensor(key))
        key2=key1.reshape(512,1)
        key3=key2.tolist()
        print(key3)
        key4=np.array(key3)

        np.savetxt(path+'itemEmb_W2.csv', key4 ,fmt='%.12f', delimiter=',')


for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(key.shape)
    if(key=='itemEmb_b2'):
        key1=reader.get_tensor(key)
        print(reader.get_tensor(key))
        key2=key1.reshape(1,1)
        key3=key2.tolist()
        print(key3)
        key4=np.array(key3)

        np.savetxt(path+'itemEmb_b2.csv', key4 ,fmt='%.12f', delimiter=',')



"""
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(key.shape)
    if (key == 'itemEmb_W3'):
        key1 = reader.get_tensor(key)
        print(reader.get_tensor(key))
        key2 = key1.reshape(64, 9)
        key3 = key2.tolist()
        print(key3)
        key4 = np.array(key3)

        np.savetxt(path + 'itemEmb_W3.csv', key4, fmt='%.12f', delimiter=',')


for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(key.shape)
    if(key=='itemEmb_b3'):
        key1=reader.get_tensor(key)
        print(reader.get_tensor(key))
        key2=key1.reshape(1,9)
        key3=key2.tolist()
        print(key3)
        key4=np.array(key3)

        np.savetxt(path+'itemEmb_b3.csv', key4 ,fmt='%.12f', delimiter=',')
"""



