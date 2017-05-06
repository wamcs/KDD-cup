import pandas as pd
import numpy as np
import split as sp
from datetime import datetime,timedelta
import split as split
import csv
import xgboost as xgb
import operator
from sklearn.svm import SVR

splitDataPath = '../dataProcessing/splitData/splitData.csv'
testDataPath = '../dataSets/testing_phase1/trajectories(table 5)_test1.csv'
testWeatherPath = '../dataSets/testing_phase1/weather (table 7)_test1.csv'
table3 = '../dataSets/training/links (table 3).csv'
table4 = '../dataSets/training/routes (table 4).csv'
table7 = '../dataProcessing/splitData/weather.csv'
# some data lack link
linkLen ={'A':{'2':6,'3':8},'B':{'1':9,'3':5},'C':{'1':12,'3':8}}

def splitTestData():
    testData,testLable = split.readData(testDataPath)
    testWData,testWLable = split.readData(testWeatherPath)
    lData,lLabel = split.readData(split.trainingPath+split.table3+split.suffix)
    linkData = split.dealLinkData(lData)
    testData,testLable = split.combineData(testData,testLable,testWData,testWLable,linkData)
    testData = np.array(testData)
    testData = pd.DataFrame(testData,columns=testLable)
    testData['starting_time'] = testData['starting_time'].apply(lambda x: x.split(' ')[0])
    testData['tollgate_id'] = testData['tollgate_id'].apply(lambda x: int(x))
    testData['time_window'] = testData['time_window'].apply(lambda x: int(x))
    return testData

def getTrainData():
    data = pd.read_csv(splitDataPath)
    data['starting_time'] = data['starting_time'].apply(lambda x: x.split(' ')[0])
    return data

def calculatedHistoryVelocityData(data):
    result = data.copy()
    result['travel_velocity'] = result['travel_velocity'].apply(lambda x: np.array(x.split(';')).astype(float))
    for intersection_id in linkLen.keys():
        for tollgate_id in linkLen[intersection_id].keys():
            index = (result['intersection_id'] == intersection_id) & (result['tollgate_id'] == int(tollgate_id))
            temp = result[:][index]
            temp = temp['vehicle_id'][temp['travel_velocity'].apply(lambda x: len(x)) != linkLen[intersection_id][tollgate_id]].index.tolist()
            result.drop(temp,inplace = True)

    result = result.drop(['vehicle_id','travel_seq'],axis = 1)
    velocitys = [['intersection_id','tollgate_id','starting_time','time_window','travel_velocity']]

    for name,group in result.groupby(['intersection_id','tollgate_id','starting_time','time_window'])['travel_velocity']:
        temp = list(name)
        temp.append(group.mean())
        velocitys.append(temp)

    velocitys = pd.DataFrame(velocitys[1:],columns=velocitys[0])

    return velocitys

def generateTrainData():
    # label = ['time','week','wind_direction','wind_speed','temperature','rel_humidity','precipitation','length','lane','objective_velocity']
    with open(split.saveLinkVelocityFilePath,'r') as fr:
        lines = csv.reader(fr)
        data = {}
        for line in lines:
            if lines.line_num == 1:
                continue
            data[line[0]] = float(line[1])

    yTrainData = []
    trainData = []
    with open(split.saveSplitDataFilePath,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            if line[8] == '':
                continue
            weather = [float(i) for i in line[10:15]]

            time = datetime.strptime(line[3], "%Y-%m-%d %H:%M:%S")
            # use 0-71 represent time_window
            time_window = (time.hour * 60 + time.minute)/20
            # 0 - 6 : Mon - Sun
            week = time.weekday()

            lanes =[int(i) for i in line[15].split(';')]
            velocitys = [float(i) for i in line[16].split(';')]
            lengths = [int(i) for i in line[17].split(';')]
            linkId = [item.split('#')[0] for item in line[4].split(';')]
            for i in range(len(lanes)):
                temp = []
                temp.append(time_window)
                temp.append(week)
                temp.extend(weather)
                temp.append(lengths[i])
                temp.append(lanes[i])
                temp.append(data[linkId[i]])
                trainData.append(temp)
                yTrainData.append(velocitys[i])
    trainData = np.array(trainData)
    yTrainData = np.array(yTrainData)
    return trainData,yTrainData



def trainVelocityRegression(xTrain,yTrain):
    # xlf = xgb.XGBRegressor(learning_rate=0.1,
    #                     n_estimators=200,
    #                     silent=1,
    #                     objective='reg:linear',
    #                     nthread=-1,
    #                     subsample=1,
    #                     colsample_bytree=0.7,
    #                     colsample_bylevel=1,
    #                     scale_pos_weight=1,
    #                     seed=1440,
    #                     missing=None)
    clf = SVR(kernel='rbf',C=1.0, epsilon=0.2,gamma=0.1,cache_size = 1000)
    print '====='
    # xlf.fit(xTrain,yTrain)
    clf.fit(xTrain,yTrain)
    print '==='
    return clf

def getCorrespondingVelocity(trainVelocity,aimVelocity):
    data = trainVelocity.copy()
    aim = aimVelocity.copy()
    data['week'] = data['starting_time'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").weekday())
    aim['week'] = aim['starting_time'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").weekday())

    trainTime = range(18,24)
    trainTime.extend(range(45,51))
    trainData = data[:][data['time_window'].isin(trainTime)]

    aim.rename(columns={'starting_time':'aim_time','travel_velocity':'aim_velocity'},inplace=True)

    trainData = pd.merge(trainData,aim,how='left',on = ['intersection_id','tollgate_id','week','time_window'])
    trainData = trainData.dropna()

    trainData['velocity_distance'] = (trainData['travel_velocity'] - trainData['aim_velocity']).apply(lambda x: np.sum(np.abs(x)))

    midData = []
    for name,group in trainData.groupby(['intersection_id','tollgate_id','aim_time','time_window']):
        minDistances = group['velocity_distance'].min()
        temp = list(name)
        temp.append(group['starting_time'][group['velocity_distance'] == minDistances].values[0])
        temp.append(minDistances)
        midData.append(temp)

    label = ['intersection_id','tollgate_id','aim_time','time_window','starting_time','velocity_distance']
    midData = pd.DataFrame(midData,columns=label)

    result = []
    aimTime = range(24,30)
    aimTime.extend(range(51,57))
    for name,group in midData.groupby(['intersection_id','tollgate_id','aim_time']):
        name = list(name)
        timeDict = {}
        dis = group['velocity_distance'].values
        for i,j in enumerate(group['starting_time']):
            if dis[i]>10:
                continue
            timeDict[j] = timeDict.get(j,0)+1

        index = (data['intersection_id'] == name[0])&(data['tollgate_id'] == name[1])&(data['starting_time'].isin(timeDict.keys()))&(data['time_window'].isin(aimTime))
        selectedData = data[['starting_time','time_window','travel_velocity']][index]

        vel = []
        for time in aimTime:
            weights = 0
            temp = selectedData[['starting_time','travel_velocity']][selectedData['time_window'] == time].values
            # some days lack data
            if temp.shape[0] == 0:
                vel.append(np.nan)
                continue

            tempVel = np.zeros(temp[0,1].shape)
            for i,startTime in enumerate(temp[:,0]):
                weight = timeDict[startTime]
                tempVel += weight*temp[i,1]
                weights += weight
            vel.append(tempVel/weights)

        for i in range(len(aimTime)):
            temp = []
            temp.extend(name)
            temp.append(aimTime[i])
            temp.append(vel[i])
            result.append(temp)

    label = ['intersection_id','tollgate_id','starting_time','time_window','his_velocity']
    return pd.DataFrame(result,columns = label)

def predictVelocity(start,end,routes,links,weatherPath,timeWindows,lackDay = []):

    weather = []
    with open(weatherPath,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            weather.append(line[:])
    weather = np.array(weather)


    startTime = datetime.strptime(start,"%Y-%m-%d")
    endTime = datetime.strptime(end,"%Y-%m-%d") + timedelta(days = 1)
    times = []
    weeks = []
    tempTime = startTime
    while tempTime != endTime:
        times.append(tempTime.strftime("%Y-%m-%d"))
        weeks.append(tempTime.weekday())
        tempTime += timedelta(days = 1)

    print '====='
    data = []
    label = ['intersection_id','tollgate_id','starting_time','link','time_window','week','wind_direction','wind_speed','temperature','rel_humidity','precipitation','length','lane','objective_velocity']
    for intersection in routes.keys():
        for tollgate in routes[intersection].keys():
            for i,time in enumerate(times):
                for window in timeWindows:
                    for link in routes[intersection][tollgate]:
                        if time in lackDay:
                            continue
                        weatherList = weather[np.nonzero((weather[:,0] == time)&(weather[:,1] == str((int(window)/9)*3)))[0],4:].astype(float).tolist()
                        if weatherList == []:
                            continue
                        temp = []
                        temp.append(intersection)
                        temp.append(tollgate)
                        temp.append(time)
                        temp.append(link)
                        temp.append(int(window))
                        temp.append(weeks[i])
                        temp.extend(weatherList[0])
                        temp.extend(links[link])
                        data.append(temp)
    data = np.array(data)

    print '====='
    trainData = data[:,4:].astype(float)
    x,y = generateTrainData()
    # wholeData = np.insert(x,0,trainData,axis=0)
    # maxVec = np.max(wholeData,axis=0).astype(float)
    # trainData = trainData/maxVec
    # x = x/maxVec
    print '====='
    xlf = trainVelocityRegression(x,y)
    print '====='
    resultVel = xlf.predict(trainData)

    data = pd.DataFrame(data,columns = label)
    data.insert(14,'predict_velocity',resultVel)

    print '====='
    result = []
    for name,group in data.groupby(['intersection_id','tollgate_id','starting_time','time_window'])['predict_velocity']:
        temp = list(name)
        velocitys = []
        for v in group:
            velocitys.append(v)
        velocitys = np.array(velocitys)
        temp.append(velocitys)
        result.append(temp)

    return pd.DataFrame(result,columns = ['intersection_id','tollgate_id','starting_time','time_window','predict_velocity'])

def predict(start,end,weatherPath,time,TestData,TrainData):
    routes = {}
    links = {}
    lengths = {}
    lengthsList = []
    with open(table3,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            # length,lane
            links[line[0]] = [int(line[1]),int(line[3])]

    with open(split.saveLinkVelocityFilePath,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            links[line[0]].append(float(line[1]))

    with open(table4,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            if line[0] not in routes.keys():
                routes[line[0]] = {}
                lengths[line[0]] = {}
            routes[line[0]][line[1]] = line[2].split(',')
            temp = []
            for i in routes[line[0]][line[1]]:
                temp.append(links[i][0])
            lengths[line[0]][line[1]] = np.array(temp)

    for i in lengths.keys():
        for j in lengths[i].keys():
            temp = []
            temp.append(i)
            temp.append(j)
            temp.append(lengths[i][j])
            lengthsList.append(temp)


    # print '==========================================='
    # print 'routes is',routes
    # print '==========================================='
    # print 'link is',links
    # print '==========================================='
    # print 'lengths is',lengths

    predictData = predictVelocity(start,end,routes,links,weatherPath,time)
    print '==========================================='
    print 'predictData is',predictData
    aimVelocity = calculatedHistoryVelocityData(TestData)
    print '==========================================='
    print 'aimVelocity:',aimVelocity
    trainVelocity = calculatedHistoryVelocityData(TrainData)
    print '==========================================='
    print 'trainVelocity',trainVelocity

    historalData = getCorrespondingVelocity(trainVelocity,aimVelocity)
    print '==========================================='
    print 'historalData',historalData
    # historalData.to_csv('../dataProcessing/log/historalData.csv')
    predictData['tollgate_id'] = predictData['tollgate_id'].apply(lambda x: int(x))
    predictData['time_window'] = predictData['time_window'].apply(lambda x: int(x))
    historalData['tollgate_id'] = historalData['tollgate_id'].apply(lambda x: int(x))
    historalData['time_window'] = historalData['time_window'].apply(lambda x: int(x))

    data = pd.merge(predictData,historalData,how = 'left', on = ['intersection_id','tollgate_id','starting_time','time_window'])
    data = data.T.fillna(method='pad', limit=1).T

    lengths = pd.DataFrame(lengthsList,columns = ['intersection_id','tollgate_id','lengths'])
    lengths['tollgate_id'] = lengths['tollgate_id'].apply(lambda x:int(x))
    data = pd.merge(data,lengths,how = 'left',on = ['intersection_id','tollgate_id'])

    preVel = data['predict_velocity'].values
    hisVel = data['his_velocity'].values
    linkLength = data['lengths'].values
    print '==========================================='
    print 'preVel',preVel
    print 'hisVel',hisVel
    print 'linkLength',linkLength
    vel = (preVel+hisVel)/2
    travelTime = []
    for i,v in enumerate(vel):
        travelTime.append(np.sum(linkLength[i]/v))

    travelTime = np.array(travelTime)

    data.insert(7,'avg_travel_time',travelTime)
    data.drop(['predict_velocity','his_velocity','lengths'],axis = 1,inplace = True)
    print '==========================================='
    print 'data is ',data
    return data

def task1():
    TrainData = getTrainData()
    TestData = splitTestData()

    print 'handle data'
    timeWindows = range(24,30)
    timeWindows.extend(range(51,57))

    print 'begain predicting'
    predictData = predict('2016-10-18','2016-10-24',testWeatherPath,timeWindows,TestData,TrainData)
    formatTrans(predictData)

def formatTrans(predictData):

    predictData['time_window'] = predictData['time_window'].apply(lambda x: int(x))
    predictData['avg_travel_time'] = predictData['avg_travel_time'].apply(lambda x: round(x,2))
    firstTime = range(24,30)
    secondTime = range(51,57)
    first = predictData[:][predictData['time_window'].isin(firstTime)]
    second = predictData[:][predictData['time_window'].isin(secondTime)]
    predictData = pd.concat([first,second])

    time = []
    days = predictData['starting_time'].values
    timeWindows = predictData['time_window'].values

    predictData.drop(['starting_time','time_window'],axis = 1,inplace = True)
    for i,day in enumerate(days):
        day = datetime.strptime(day,"%Y-%m-%d")
        minute = int(timeWindows[i])*20
        hour = minute/60
        minute = minute - hour*60
        day = day.replace(hour = hour,minute = minute)
        start = day
        end = day + timedelta(minutes = 20)
        time.append("["+start.strftime("%Y-%m-%d %H:%M:%S")+","+end.strftime("%Y-%m-%d %H:%M:%S")+")")
    time = np.array(time)
    predictData.insert(2,'time_window',time)
    predictData.to_csv('../dataProcessing/answerSVR.csv',index = False)



if __name__ == '__main__':
    task1()
