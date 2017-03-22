# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

path = '../dataProcessing/splitData/weather.csv'
savePath = '../dataProcessing/weatherRelation/'

# fontSize
fs = 18
# markersize
ms = 10
Xrange = [(900,1100),(900,1100),(0,360),(0,10),(0,50),(0,100),(0,30)]

def readData(filename):
    data = []
    with open(filename,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            data.append(line)
    return data

def main():

    data = readData(path)
    lables = data[0]
    lables = lables[2:]
    data.remove(data[0])
    data = np.array(data)
    data = data[:,2:].astype(float)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for lable1 in lables[2:]:
        for lable2 in lables[2:]:
            if lables.index(lable1)>=lables.index(lable2):
                continue

            pltFile = savePath+lable1+'_'+lable2+'.png'
            weather1 = data[:,lables.index(lable1)]
            weather2 = data[:,lables.index(lable2)]

            plt.plot(weather1,weather2,'bx',markersize = ms)
            plt.title(lable1+'&'+lable2,fontsize = fs)
            plt.xlim(Xrange[lables.index(lable1)])
            plt.ylim(Xrange[lables.index(lable2)])
            plt.xlabel(lable1,fontsize = fs)
            plt.ylabel(lable2,fontsize = fs)
            plt.savefig(pltFile)
            plt.close()




if __name__ == '__main__':
    main()
