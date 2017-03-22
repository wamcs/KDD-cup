# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

dataPath = '../dataProcessing/linkVolecity/LinkVolecity_'
dataSuffix = '.csv'
table3 = '../dataSets/training/links (table 3).csv'

saveFigurePath = '../dataProcessing/vecolity_figure/'
figureSuffix = '.png'

# fontSize
fs = 18
# markersize
ms = 10

def readData(filename):
    with open(dataPath+filename+dataSuffix,'r') as fr:
        lines = csv.reader(fr)
        data = []
        for line in lines:
            data.append(line[:])
        data.remove(data[0])
    return data

def getLinkList():
    result = []
    with open(table3,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            result.append(line[0])
        result.remove(result[0])
    return result


def main():
    if not os.path.exists(saveFigurePath):
        os.makedirs(saveFigurePath)
    linkList = getLinkList()
    for link in linkList:
        data = readData(link)
        data = np.array(data)

        maxV = (max([float(x) for x in data[:,1]])/5+1)*5
        minV = (min([float(x) for x in data[:,1]])/5-1)*5
        # print data
        for i in range(7):
            weekData = data[np.nonzero(data[:,-1] == str(i))[0],:]

            pltFile = saveFigurePath+link+'_'+str(i)+figureSuffix
            plt.plot(weekData[:,0],weekData[:,1],'b--',linewidth=1)
            plt.title('link'+link+'&velocity',fontsize = fs)
            plt.xlim((0,72))
            plt.ylim((minV,maxV))
            plt.xlabel('week'+str(i),fontsize = fs)
            plt.ylabel('velocity',fontsize = fs)
            plt.savefig(pltFile)
            plt.close()

if __name__ == '__main__':
    main()
