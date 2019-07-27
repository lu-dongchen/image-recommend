


import cv2
from pylab import *


path="user/user0/test/"

#img1 = cv2.imread('user/test0_pos/f20baa776c83f243b9e5bc8441445e93.jpg')
img1 = cv2.imread(path+'pos/329c7b3054ab20196696596e251ce40b.jpg')
img2 = cv2.imread(path+'pos/1954bc891529ce6583d50f29902ffad9.jpg')
img3 = cv2.imread(path+'neg/3f390adc52ea0232ca4d3a1822237e29.jpg')
#img4 = cv2.imread('user/user0_pos/ef9bd17d3ec364452939237cda43be98.jpg')
img5 = cv2.imread('user/user0_pos/f27894d1f905ce748944b13eaaa9c1fe.jpg')
img6 = cv2.imread('user/user0_pos/f76f914f9bff996fae9e4e171ef1bb25.jpg')

# 将opencv中的BGR、GRAY格式转换为RGB，使matplotlib中能正常显示opencv的图像

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
#img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
#img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
#img5 = cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)

fig = plt.figure()
subplot(131)
imshow(img1)
title('rec1')
axis('off')
subplot(132)
imshow(img2)
title('rec2')
axis('off')
subplot(133)
imshow(img3)
title('rec3')
axis('off')
#subplot(154)
#imshow(img3)
#title('rec4')
#axis('off')
#subplot(155)
#imshow(img3)
#title('rec5')
#axis('off')
#subplot(166)
#imshow(img6)
#title('rec5')
#axis('off')
show()


##plt 同时显示多幅图像


"""
import matplotlib.pyplot as plt

plt.figure()

img1 = plt.imread('user/test0_pos/f20baa776c83f243b9e5bc8441445e93.jpg')
img2 = plt.imread('user/test0_pos/f91063e3de10b7835a7179c935899b1e.jpg')
img3 = plt.imread('user/test0_pos/f66340cf64c27e5ea236375f9fdd981e.jpg')


plt.subplot(1, 3, 1)

plt.imshow(img1)

plt.subplot(1, 3, 2)

plt.imshow(img3)

plt.subplot(1, 3, 3)

plt.imshow(img2)

plt.show()

"""