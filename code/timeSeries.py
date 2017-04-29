
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as st
import pyflux as pf
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import os

splitDataPath = '../dataProcessing/splitData/splitData.csv'
testDataPath = '../dataProcessing/splitData/testData.csv'
testWeatherPath = '../dataSets/testing_phase1/weather (table 7)_test1.csv'
table3 = '../dataSets/training/links (table 3).csv'
table4 = '../dataSets/training/routes (table 4).csv'
table7 = '../dataProcessing/splitData/weather.csv'
arimaPicturePath = '../dataProcessing/arimaPicture/'

def getData(path):
    data = pd.read_csv(path)
    return data

def regularTime(s):
    time = datetime.strptime(s,"%Y-%m-%d %H:%M:%S")
    minute = time.minute
    time = time.replace(minute = (minute/20)*20,second = 0)
    return time.strftime("%Y-%m-%d %H:%M:%S")

def splitDataByLane(data,routes):
#     label = ['intersection_id','tollgate_id','lane_id','starting_time','time_window','week','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation','travel_lane','travel_velocity','travel_length']
    label = ['lane_id','time','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation','travel_velocity']
    dataSet = data.copy()
    dataSet['travel_velocity'] = dataSet['travel_velocity'].apply(lambda x: np.array(x.split(';')).astype(float))
    dataSet['travel_lane'] = dataSet['travel_lane'].apply(lambda x: np.array(x.split(';')).astype(float))
    dataSet['travel_length'] = dataSet['travel_length'].apply(lambda x: np.array(x.split(';')).astype(float))
    dataSet['travel_seq'] = dataSet['travel_seq'].apply(lambda x: x.split(';'))
    dataSet['starting_time'] = dataSet['starting_time'].apply(regularTime)
    dataSet.fillna(-1)

    dataSet.drop(['vehicle_id','total_velocity','total_length'],axis = 1,inplace = True)


    weather = []
    weaTmp = dataSet[['starting_time','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation']]
    for name,group in weaTmp.groupby(['starting_time']):
        temp = []
        temp.extend(group.values[0].tolist())
        weather.append(temp)
    wLabel = ['time','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation']
    weather = pd.DataFrame(weather,columns = wLabel)

    print '==========================='
    print 'weather data is'
    print weather.info()

    result = pd.concat(list(dataSet['travel_seq'].apply(lambda x: pd.DataFrame(x,columns=['lane_id']))))
    result['time'] = result['lane_id'].apply(lambda x: x.split('#')[1])
    result['time'] = result['time'].apply(regularTime)
    result['lane_id'] = result['lane_id'].apply(lambda x: x.split('#')[0])
    result['travel_velocity'] = pd.concat(list(dataSet['travel_velocity'].apply(lambda x: pd.DataFrame(x))))
    result = pd.merge(result,weather,how = 'left',on = ['time'])

    result['time'] = result['time'].apply(pd.to_datetime)
    result.set_index('time',inplace = True)

    print '==========================='
    print 'begain to split'
    print result.info()

    dataByLink = {}

    for intersection in routes.keys():
        for tollgate in routes[intersection].keys():
            for lane in routes[intersection][tollgate]:
                temp = result[:][result['lane_id'] == lane]
                dataByLink[lane] = temp.drop(['lane_id'],axis = 1)
#                 print '==========='
#                 print lane
#                 print dataByLink[lane].info()


    return dataByLink

def handleWeaInflu(data,isTest = False):
    label = ['travel_velocity']
    # some day (9-28,10-10) lack the information of weather
    handleData = data.copy()

    handleData = handleData['travel_velocity'].to_frame()

    # ga
    if not isTest:
        time=pd.DataFrame(pd.date_range(start = '20160719000000',end = '20161017234000',freq = '20min'),columns=['time'])
        result = pd.merge(time,handleData,how='left',left_on='time',right_index=True)
        result.set_index('time',inplace = True)
        result = result.interpolate()
    else:
        result = handleData

    return result

# def recoverWea(data):


def predict():
    routes = {}
    links = {}
    with open(table3,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            # length,lane
            links[line[0]] = [int(line[1]),int(line[3])]

    with open(table4,'r') as fr:
        lines = csv.reader(fr)
        for line in lines:
            if lines.line_num == 1:
                continue
            if line[0] not in routes.keys():
                routes[line[0]] = {}
            routes[line[0]][line[1]] = line[2].split(',')

    print '==========================='
    print 'preparation completes'

    trainData = getData(splitDataPath)
    testData = getData(testDataPath)
    trainDataByLink = splitDataByLane(trainData,routes)
    testDataByLink = splitDataByLane(testData,routes)
    predictLaneVelo = {}


    print '==========================='
    print 'prepate with lane'
    if not os.path.exists(arimaPicturePath):
        os.makedirs(arimaPicturePath)

    for lane in trainDataByLink.keys():
        print '==========================='
        print lane
        data = trainDataByLink[lane]
        testData = handleWeaInflu(testDataByLink[lane],True)
        handleData = handleWeaInflu(data)
        handleData.plot()
        result = []
        plt.savefig(arimaPicturePath+lane+'_trace_picture.png')
        test_span = 6
        size = 14
        i = 0
        while(i<size):
            predict = run_aram(handleData,10,10,18)
            handleData = pd.concat([handleData,predict,testData[i*6:(i+1)*6]])
            print handleData
            if i%2!=0:
                result.append(predict[0:6])
                predict = run_arm(handleData,10,10,24)
                handleData = pd.concat([handleData,predict])
                result.append(predict[0:6])
            i = i+1
        temp = pd.concat(result)
        predictLaneVelo[lane] = temp

    for lane in links.keys():
        # recover weather
        predictLaneVelo[lane]['time'] = float(links[lane][0])/predictLaneVelo[lane]['travel_velocity']

    result = []
    for intersection in routes.keys():
        for tollgate in routes[intersection].keys():
            temp = predictLaneVelo[lane].copy()
            temp['intersection'] = intersection
            temp['tollgate'] = tollgate
            temp.drop(['time','travel_velocity'],axis = 1,inplace = True)
            temp['predict_time'] = temp.index
            temp['predict_name'] = 0
            for lane in routes[intersection][tollgate]:
                temp['predict_time'] = temp['predict_time']+predictLaneVelo[lane]['time']
            result.append(temp)


    result = pd.concat(result)
    result.to_csv(arimaPicturePath+'result.csv')



def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    print dftest
    return dftest[1]

def bestDiff(df, maxdiff = 8):
    p_set = {}
    for i in range(0, maxdiff):
        temp = df.copy() #reset before recycle
        if i == 0:
            temp['diff'] = temp[temp.columns[1]]
        else:
            temp['diff'] = temp[temp.columns[1]].diff(i)
            temp = temp.drop(temp.iloc[:i].index) #after diffing, front data will be nan ,so delete it
        pvalue = test_stationarity(temp['diff'])
        p_set[i] = pvalue
        p_df = pd.DataFrame.from_dict(p_set, orient="index")
        p_df.columns = ['p_value']
    i = 0
    while i < len(p_df):
        if p_df['p_value'][i]<0.01:
            bestdiff = i
            break
        i += 1
    return bestdiff

def produce_diffed_timeseries(df, diffn):
    if diffn != 0:
        df['diff'] = df[df.columns[1]].apply(lambda x:float(x)).diff(diffn)
    else:
        df['diff'] = df[df.columns[1]].apply(lambda x:float(x))
    df.dropna(inplace=True) #差分之后的nan去掉
    return df

def choose_order(ts, maxar, maxma):
    print 'choose order'
    order = st.arma_order_select_ic(ts, maxar, maxma, ic=['aic', 'bic', 'hqic'])
    print 'finish'
    return order.bic_min_order

def predict_recover(ts, df, diffn):
    if diffn != 0:
        ts.iloc[0] = ts.iloc[0]+df['log'][-diffn]
        ts = ts.cumsum()
    ts = np.exp(ts)
#    ts.dropna(inplace=True)
    print('还原完成')
    return ts


def run_aram(df, maxar, maxma, test_size = 14):

    train = df.dropna()
    train['log'] = np.log(train[train.columns[0]])

    if test_stationarity(train[train.columns[1]]) < 0.01:
        train['diff'] = train['log']
        print('平稳，不需要差分')
    else:
        diffn = best_diff(train, maxdiff = 8)
        train = produce_diffed_timeseries(train, diffn)
        print('差分阶数为'+str(diffn)+'，已完成差分')
    print('开始进行ARMA拟合')
    print train[train.columns[2]]
    order = choose_order(train[train.columns[2]], maxar, maxma)
    print('模型的阶数为：'+str(order))
    _ar = order[0]
    _ma = order[1]
    model = pf.ARIMA(data=train, ar=_ar, ma=_ma, target='diff', family=pf.Normal())
    model.fit("MLE")
    test_predict = model.predict(int(test_size))
    test_predict = predict_recover(test_predict, train, diffn)
    return test_predict


if __name__ == '__main__':
    predict()
