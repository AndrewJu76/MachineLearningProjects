import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from collections import Counter

# img = image.open('starbucks.jpg')
# print(img.size,img.mode,img.format)

def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = image.open(f)
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    return data,m,n

data,row,col = loadData('starbucks.jpg')
# print(data,m,n)
imgData = np.array(data)
# print(imgData.shape)
label = KMeans(n_clusters=4).fit_predict(imgData)
res = Counter(label)
# print(dict(res))

label = label.reshape([row,col])
print(label.shape)
pic_new = image.new("L", (row, col))
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j), int(256/(label[i][j]+1)))
pic_new.save("result-bull-4.jpg", "JPEG")