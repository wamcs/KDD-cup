# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

dataPath = '../dataProcessing/splitData/splitData_'
dataSuffix = '.csv'
dataList = ['A2','A3','B1','B3','C1','C3']

savePath = '../dataProcessing/weather_figure/'
figureSuffix = '.png'

Xrange = [(900,1100),(900,1100),(0,360),(0,10),(0,50),(0,100),(0,30)]

# fontSize
fs = 18
# markersize
ms = 10

def readData(filename):
    data = []
    with open(dataPath+filename+dataSuffix,'r') as fr:
        lines = csv.reader(fr)
        for i,line in enumerate(lines):
            # ignore first row and those rows lacking weather information
            if i == 0:
                temp = line[6:13]
                temp.append(line[17])
                data.append(temp)
                continue
            if line[6] == '':
                continue
            temp = [float(x) for x in line[6:13]]
            temp.append(float(line[17]))
            data.append(temp)
        # print i
    return data


def main():
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    allData = []

    for i in dataList:
        data = readData(str(i))
        # print len(data)
        # print i
        lables = data[0][0:-1]
        data.remove(data[0])
        allData.extend(data)
        dataArray = np.array(data)
        # print lables
        velocity = dataArray[:,-1]
        for lable in lables:

            pltFile = savePath+lable+'_'+str(i)+figureSuffix
            weather = dataArray[:,lables.index(lable)]

            plt.plot(weather,velocity,'bx',markersize = ms)
            plt.title(lable+'&velocity_'+i,fontsize = fs)
            plt.xlim(Xrange[lables.index(lable)])
            plt.xlabel(lable,fontsize = fs)
            plt.ylabel('velocity',fontsize = fs)
            plt.savefig(pltFile)
            plt.close()

    allData = np.array(allData)
    # print allData
    for lable in lables:
        pltFile = savePath+lable+figureSuffix
        weather = allData[:,lables.index(lable)]
        velocity = allData[:,-1]
        plt.plot(weather,velocity,'bx',markersize = ms)
        plt.title(lable+'&velocity',fontsize = fs)
        plt.xlim(Xrange[lables.index(lable)])
        plt.xlabel(lable,fontsize = fs)
        plt.ylabel('velocity',fontsize = fs)
        plt.savefig(pltFile)
        plt.close()





if __name__ == '__main__':
    main()
