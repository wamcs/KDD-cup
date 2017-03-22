# -*- coding utf-8 -*-

import numpy as np
import csv
import operator
from sklearn.neighbors import NearestNeighbors

path1 = '../dataProcessing/splitData/weather.csv'
path2 = '../dataProcessing/splitData/splitData.csv'
path3 = '../dataProcessing/linkVolecity/LinkVolecity_'

def readData(filename):
    data = []
    with open(filename,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            data.append(line)
    return data

def getLinkList():
    table3 = '../dataSets/training/links (table 3).csv'
    result = []
    with open(table3,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            result.append(line[0])
        result.remove(result[0])
    return result

def calculateCov(data):
    index = [2,3,4,5,6,7,8]
    m = len(index)
    n = len(data)
    partData = []
    label = data[0][2:]


    for i in range(n):
        if i == 0:
            continue
        temp = []
        for j in index:
            temp.append(float(data[i][j]))
        partData.append(temp)

    partData = np.array(partData)
    average = []
    m,n = np.shape(partData)

    for i in range(n):
        average.append((np.sum(partData[:,i])/m)*np.ones((1,m)))

    for i in range(n):
        average[i] = partData[:,i] - average[i]


    result = []
    for i in range(n):
        temp = []
        for j in range(n):
            temp.append(np.sum(average[i]*average[j])/m)
        result.append(temp)

    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j]>50 or result[i][j]<-50:
                print 'position is (%s,%s) and coefficient is %d' %(label[i],label[j],int(result[i][j]))
    print label
    print np.mat(result)
    print np.linalg.eig(np.mat(result))


def getObjectVelocity(vData,ivList,tMode,hMode):
    data = np.array(vData)
    indexList = np.array(ivList)[:,0].astype(np.int64)

    # print np.shape(indexList)
    # print np.shape(data)
    data = data[indexList,:]

    m,n = np.shape(data)
    # wind_direction;temperate,humidity,preciption
    wList = np.zeros((m,4))
    for i in range(m):
        for j in range(n):
            if j>=9 and j<=12:
                if data[i][j] == '':
                    # over normal number
                    wList[i][j-9] = 10000
                    continue
                wList[i][j-9] = float(data[i][j])

    indexs = np.nonzero((wList[:,0] <= 1.5) & (wList[:,3] == 0))[0]
    print indexs


    idealWeather = np.array([tMode,hMode])


    # use knn choose the best weather
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(wList[indexs,1:3])
    distances,indices = nbrs.kneighbors(idealWeather)
    # print indices

    best = np.array(ivList)[indexs[indices],1]
    print data[indexs[indices]]
    print wList[indexs[indices]]
    print ivList[indexs[indices]]
    return np.max(best)




def getLinkVelocity():

    # vData = readData(path2)
    # vData.remove(vData[0])
    # wData = readData(path1)
    # wData.remove(wData[0])
    # wData = np.array(wData)
    # # tMode = calculateMode(wData[:,6])
    # # hMode = calculateMode(wData[:,7])
    # tMode = np.mean(wData[:,6].astype(float))
    # hMode = np.mean(wData[:,7].astype(float))
    #
    # print tMode
    # print hMode

    result = {}
    linkList = getLinkList()
    for link in linkList:
        print link
        data = readData(path3+link+'.csv')
        data.remove(data[0])
        result[link] = []
        for item in data:
            index = item[2].split(';')
            vel = item[3].split(';')
            vec = [[int(index[i]),float(vel[i])] for i in range(len(index))]
            result[link].extend(vec)
        # result[link] = sorted(result[link],key = operator.itemgetter(0))
        # velocity = getObjectVelocity(vData,result[link],float(tMode),float(hMode))
        # result[link] = velocity
        print sorted(result[link],key = operator.itemgetter(1))[-1][0]

        result[link] = sorted(result[link],key = operator.itemgetter(1))[-1][1]

    LinkVolecity = [['linkId','volecity']]
    for key in result.keys():
        LinkVolecity.append([key,result[key]])

    np.savetxt('../dataProcessing/linkVolecity/LinkVolecity.csv',np.array(LinkVolecity),delimiter=',',fmt='%s')


# #  calculate mode
# def calculateMode(vec):
#     mode = {}
#     for i in vec:
#         if i not in mode.keys():
#             mode[i] = 0
#         else:
#             mode[i]+=1
#
#     sortedMode = sorted(mode.iteritems(),key = operator.itemgetter(1),reverse=True)
#     return sortedMode[0][0]


# test function:
def main():

    wData = readData(path1)
    vData = readData(path2)

    calculateCov(wData)
    # getLinkVelocity()

if __name__ == '__main__':
    main()
