# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5']
x = range(len(names))
y = [3.64, 3.30, 3.02, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]
y1 = [4.30, 4.30, 4.30, 4.30, 4.30,4.30, 0, 0, 0, 0 ]
y2 = [0.67,0.70,0.71,0.68,0.71]

y_1=[0.0178,0.0475,0.0825,0.1580,0.3505]
y1_1=[0.0178,0.0555,0.0937,0.1775,0.3618]
y2_1=[0.0198,0.0573,0.1038,0.2051,0.3952]

# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 20) # 限定横轴的范围
plt.ylim(0, 5) # 限定纵轴的范围
plt.plot(x, y,  mec='r', mfc='w', label='TTL与非门')
plt.plot(x, y1, ms=10, label='CMOS与非门')
#plt.plot(x, y2_1, marker='^', ms=10, label='MF-bpr')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Ui")  # X轴标签
plt.ylabel("Uo")  # Y轴标签
plt.title("特性曲线")  # 标题

plt.show()