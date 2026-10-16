import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    fr = open(filePath,'r+',encoding='gbk')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName

data,cityName = loadData('city.txt')
# print(cityName)
# print(data)
km = KMeans(n_clusters=4)
label = km.fit_predict(data)
expenses = np.sum(km.cluster_centers_,axis=1)
# print(expenses)
# print(km.cluster_centers_)
# print(km.cluster_centers_.shape)
# print(label)
# for i in range(len(cityName)):
#     print(cityName[i])
#     print(label[i])
CityCluster = [[], [], [],[]]
for i in range(len(cityName)):
    CityCluster[label[i]].append(cityName[i])
for i in range(len(CityCluster)):
    print("Expenses:%.2f" % expenses[i])
    print(CityCluster[i])