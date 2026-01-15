import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


data = pd.read_csv('.\\Samples\\USA_Housing.csv')


X = data.drop(['Price'], axis=1)
y = data.loc[:, 'Price']

print(X.shape, y.shape)

LR_multi = LinearRegression()
LR_multi.fit(X, y)

# make prediction
y_pred_multi = LR_multi.predict(X)

mean_absolute_error = mean_squared_error(y, y_pred_multi)
r2_score = r2_score(y, y_pred_multi)
print('Mean Absolute Error:', mean_absolute_error)
print('R2 Score:', r2_score)

fig = plt.figure(figsize=(8,8))
fig1 = plt.subplot(231)
plt.scatter(data['Avg. Area Income'], data['Price'])
plt.title('Price Vs Income')

fig2 = plt.subplot(232)
plt.scatter(data['Avg. Area House Age'], data['Price'])
plt.title('Price Vs House Age')

fig3 = plt.subplot(233)
plt.scatter(data['Avg. Area Number of Rooms'], data['Price'])
plt.title('Price Vs Rooms')

fig4 = plt.subplot(234)
plt.scatter(data['Avg. Area Number of Bedrooms'], data['Price'])
plt.title('Price Vs Bedrooms')

fig5 = plt.subplot(235)
plt.scatter(data['Area Population'], data['Price'])
plt.title('Price Vs Population')

fig6 = plt.subplot(235)
plt.scatter(y, y_pred_multi)
plt.title('Real Price Vs Predicted Price ')
plt.show()


# X = data.loc[:, 'Avg. Area Income']
# y = data.loc[:, 'Price']
# X = np.array(X).reshape(-1, 1)
# # print(X.shape, y.shape)
# LR1 = LinearRegression()
# LR1.fit(X, y)
#
# y_predict_1 = LR1.predict(X)
# mean_absolute_error_1 = mean_squared_error(y, y_predict_1)
# r2_score_1 = r2_score(y, y_predict_1)
# # print('Mean Absolute Error:', mean_absolute_error_1)
# # print('R2 Score:', r2_score_1)



