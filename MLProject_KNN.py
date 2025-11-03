import numpy as np
import collections as c

# feature:[120,138,156...]
def knn(k,predictPoint,feature,label):
    distance = list(map(lambda x:abs(predictPoint-x),feature))
    sortindex = np.argsort(distance)
    return c.Counter(label[sortindex][0:k]).most_common(1)[0][0]

# feature:[[120,0.5],[138,0.53],[156,0.55]...]
def knn2(k,predictPoint,ballcolor,feature,label):
    distance = list(map(lambda item: ((item[0] - predictPoint) ** 2 + (item[1] - ballcolor) ** 2) ** 0.5, feature))
    sortindex = np.argsort(distance)
    return c.Counter(label[sortindex][0:k]).most_common(1)[0][0]

# Normalization
def knn3(k,predictPoint,ballcolor,feature,label):
    distance = list(map(lambda item:((item[0]/475-predictPoint/475)**2+((item[1]-0.50)/0.05-(ballcolor-0.50)/0.05)**2)**0.5,feature))
    sortindex = np.argsort(distance)
    return c.Counter(label[sortindex][0:k]).most_common(1)[0][0]

def color2num(str):
    dict = {'红':0.5,'黄':0.51,'蓝':0.52,'绿':0.53,'紫':0.54,'粉':0.55}
    return dict[str]

if __name__ == '__main__':
    traindata = np.loadtxt('.\\Samples\\data2-train.csv',delimiter=',')
    testdata = np.loadtxt('.\\Samples\\data2-test.csv', delimiter=',')
    # feature = traindata[:,0]
    feature = traindata[:, 0:2]
    label = traindata[:,-1]

    k=36
    count = 0
    for item in testdata:
        # predict = knn(k, item[0], feature, label)
        predict = knn3(k, item[0],item[1],feature, label)
        real = item[-1]
        if predict == real:
            count = count + 1
    print('k={},准确率:{}%'.format(k,count*100/len(testdata)))


