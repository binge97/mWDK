import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter   ### 今天的主角
def formatnum(x, pos):
    return '$10^{%d}$' % (x)


# plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
#折线图
x = [3,4,5,6]#点的横坐标
# x=[1000,10000,100000,1000000]

formatter1 = FuncFormatter(formatnum)
f, ax = plt.subplots(1, 1)
ax.xaxis.set_major_formatter(formatter1)
ax.yaxis.set_major_formatter(formatter1)
edg =  [121,13816,140923,18694195]
iik = [0.29,	3.97,	49.65,	572.13]#线1的纵坐标
wl = [0.01,0.13,   3.02,   50.56,]#线2的纵坐标
ik = [0.27, 4.01, 59.30, 680.05]
daegc = [47,121.37, 1547.22,0]
NDLS=[0.11,2.35,92.8,1642.5,]
iik = np.log10(iik)
ik = np.log10(ik)
wl = np.log10(wl)
NDLS = np.log10(NDLS)
daegc = np.log10(daegc)


plt.plot(x,wl,'s-',color = 'b',label="WL")#s-:方形
plt.plot(x,ik,'o-',color = 'g',label="WDK")#o-:圆形
plt.plot(x,iik,'o-',color = 'r',label="mWDK")#o-:圆形
plt.plot(x,daegc,'s-',color = 'y',label="DAEGC")#s-:方形
plt.plot(x,NDLS,'o-',color = 'k',label="NDLS")#o-:圆形


# plt.xlabel()#横坐标名字
plt.ylabel("Time / second")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()