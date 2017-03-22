path = '../dataSets/training/'
table3 = 'links (table 3)'
table5 = 'trajectories(table 5)_training'
table7 = 'weather (table 7)_training'

suffix = '.csv'

import csv
from datetime import datetime
import numpy as np
import matplotlib as pl
import math

# read data from file
def readData(table):
    filename = path+table+suffix
    fr = open(filename,'r')
    lines = csv.reader(fr)
    data = []
    for line in lines:
        data.append(line[:])
    labelVec = data[0]
    data.remove(data[0])
    return data,labelVec

# combine weather data,link data and some calculated result with trajectories data,and split id with intersection_id and tollgate_id
# lack weather data from 9.1-9.10
def combineData(vData,vLabel,wData,wLabel,lData):
    label = vLabel
    label.extend(wLabel[2:])

    # trace_width:spilt link width with notation ';'
    # trace_length:like among
    # trace_velocity:like among
    newLabel = ['travel_lane','travel_velocity','travel_length','total_length','total_velocity']
    # newLabel = ['travel_lane','travel_velocity','travel_length','total_length','total_velocity','affacted_objective_velocity','predict_link_time','predict_total_time']
    label.extend(newLabel)

    #wData wind_direction has some data over range
    for i in range(len(wData)-1):
        if float(wData[i+1][4])>=360:
            wData[i+1][4] = '0.0000'



    # result date
    data = []
    temp = []
    i = 0
    # for complementing the lack weather data
    blank = ['','','','','','','']
    # for leaping the interval of the time of lack weather
    gap = False

    date = datetime.strptime(wData[i][0],"%Y-%m-%d")
    hour = int(wData[i][1])

    for item in vData:


        # add weather information
        time = datetime.strptime(item[3], "%Y-%m-%d %H:%M:%S")
        # ignore National Day and Mid-Autumn
        if (time.month == 10 and time.day>=1 and time.day<=7) or (time.month == 9 and time.day <=16 and time.day >=14):
            continue

        temp = item
        gap = False
        while not (time.year == date.year and time.month == date.month and time.day == date.day and (int(time.hour)/3)*3 == hour):
            if (time.month == date.month and time.day < date.day) or (time.month < date.month) or (time.month == date.month and time.day == date.day and (int(time.hour)/3)*3 < hour):
                gap = True
                break

            i=i+1
            date = datetime.strptime(wData[i][0],"%Y-%m-%d")
            hour = int(wData[i][1])


        if not gap:
            temp.extend(wData[i][2:])
        else:
            temp.extend(blank)

        # add link information and velocity
        travel_lane = []
        travel_velocity = []
        travel_length = []
        total_length = 0
        total_velocity = 0
        affacted_objective_velocity = []
        predict_link_time = []

        travel_seq = item[4].split(';')
        for seqItem in travel_seq:
            section = seqItem.split('#')
            linkId = section[0]
            itemTime = section[2]

            travel_lane.append(lData[linkId][1])
            travel_length.append(lData[linkId][0])
            itemVelocity = float(lData[linkId][0])/float(itemTime)
            travel_velocity.append(str(itemVelocity))
            total_length += int(lData[linkId][0])
        total_velocity = float(total_length)/float(item[5])

        temp.append(';'.join(travel_lane))
        temp.append(';'.join(travel_velocity))
        temp.append(';'.join(travel_length))
        temp.append(total_length)
        temp.append(total_velocity)

        data.append(temp)

    return data,label

# transfer link data to dictionary
def dealLinkData(lData):
    result = {}
    for i in lData:
        result[i[0]] = [i[1],i[3]]
    return result

def printLinkVolecity(data,lData):
    result = {}
    final = {}
    for item in lData:
        result[item[0]] = []
        final[item[0]] = [['time','velocity','informationId','partVelocity','week']]
    for i,item in enumerate(data):
        travel_seq = item[4].split(';')
        travel_velocity = item[14].split(';')
        for j,seqItem in enumerate(travel_seq):
            section = seqItem.split('#')
            linkId = section[0]
            temp = [item[3],float(travel_velocity[j]),i]
            result[linkId].append(temp)

    for linkId in result.keys():
        travel_times = {}
        # {week:{time_window:{v1,v2...}}}
        for item in result[linkId]:
            time = item[0]
            time = datetime.strptime(time,"%Y-%m-%d %H:%M:%S")

            time_window = int(math.floor(time.minute / 20) * 20)
            # use 0-71 represent time_window
            time_span = (time.hour * 60 + time_window)/20

            week = time.weekday()

            if week not in travel_times.keys():
                travel_times[week] = {}

            if time_span not in travel_times[week].keys():
                travel_times[week][time_span] = [item[1:]]
            else:
                travel_times[week][time_span].append(item[1:])
        # print travel_times
        weeks = list(travel_times.keys())
        # print weeks
        for w in weeks:
            travel_times_week = travel_times[w]
            for span in travel_times_week.keys():
                length = len(travel_times_week[span])
                length = length/2
                velocityList = []
                indexList = []

                for item in travel_times_week[span]:
                    velocityList.append(item[0])
                    indexList.append(item[1])

                velocity = sum(velocityList)/len(velocityList)

                velocitySeq = ';'.join([str(i)  for i in velocityList])
                indexSeq = ';'.join([str(i)  for i in indexList])
                temp = [span,velocity,indexSeq,velocitySeq,w]
                final[linkId].append(temp)

    for item in final.keys():
        temp = final[item]
        temp = np.array(temp)
        np.savetxt('../dataProcessing/linkVolecity/LinkVolecity_'+str(item)+'.csv',temp,delimiter=',',fmt='%s')


def dealVData(data):
    data = np.array(data)
    A = data[np.nonzero(data[:,0] == 'A')[0],:]
    B = data[np.nonzero(data[:,0] == 'B')[0],:]
    C = data[np.nonzero(data[:,0] == 'C')[0],:]
    A2 = A[np.nonzero(A[:,1]== '2')[0],:]
    A3 = A[np.nonzero(A[:,1]== '3')[0],:]
    B1 = B[np.nonzero(B[:,1]== '1')[0],:]
    B3 = B[np.nonzero(B[:,1]== '3')[0],:]
    C1 = C[np.nonzero(C[:,1]== '1')[0],:]
    C3 = C[np.nonzero(C[:,1]== '3')[0],:]
    data = []
    data.append(A2)
    data.append(A3)
    data.append(B1)
    data.append(B3)
    data.append(C1)
    data.append(C3)
    return data


def printSplitData(result,label):

    for i in range(len(result)):
        temp = np.insert(result[i],0,label,axis=0)
        np.savetxt('../dataProcessing/splitData/splitData_'+str(temp[1][0])+str(temp[1][1])+'.csv',temp,delimiter=',',fmt="%s")

def printCombineData(data,label):
    result = np.array(data)
    result = np.insert(result,0,label,axis=0)
    np.savetxt('../dataProcessing/splitData/splitData.csv',result,delimiter=',',fmt='%s')

def printRegularWeather(wData,wLabel):
    for i in range(len(wData)-1):
        if float(wData[i+1][4])>=360:
            wData[i+1][4] = '0.0000'
    result = np.array(wData)
    temp = np.insert(wData,0,wLabel,axis=0)
    np.savetxt('../dataProcessing/splitData/weather.csv',temp,delimiter=',',fmt='%s')

def main():
    lData,lLabel = readData(table3)
    vData,vLabel = readData(table5)
    wData,wLabel = readData(table7)

    linkData = dealLinkData(lData)
    data,label = combineData(vData,vLabel,wData,wLabel,linkData)
    result = dealVData(data)

    printCombineData(data,label)
    printSplitData(result,label)
    printLinkVolecity(data,lData)
    printRegularWeather(wData,wLabel)

if __name__ == '__main__':
    main()
