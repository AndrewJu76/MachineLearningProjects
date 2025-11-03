import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# multiply linear regression
# y = w1#x1+w2#x2+w3#x3...+b
url = ".\\Samples\\Salary_Data2.csv"
data = pd.read_csv(url)

# Label Encoding
data['EducationLevel'] = data['EducationLevel'].map({'高中以下':0,'大學':1,'碩士以上':2})

# One-Hot Encoding
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(data[['City']])
city_encoded = onehot_encoder.transform(data[['City']]).toarray()

data[['CityA','CityB','CityC']] = city_encoded
data = data.drop(['City','CityC'],axis=1)

x = data[['YearsExperience','EducationLevel','CityA','CityB']]
y = data['Salary']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=87)
x_train =x_train.to_numpy()
x_test =x_test.to_numpy()

# standardization
scarler = StandardScaler()
scarler.fit(x_train)
x_train = scarler.transform(x_train)
x_test = scarler.transform(x_test)

# y_pred = (x_train*w).sum(axis=1) + b
def computer_cost(x,y,w,b):
    y_pred = (x * w).sum(axis=1) + b
    cost = ((y-y_pred)**2).mean()
    return cost

def computer_gradient(x, y, w, b):
    y_pred = (x * w).sum(axis=1) + b
    w_gradient = np.zeros(x.shape[1])
    b_gradient = ((y_pred-y)).mean()
    for i in range(x.shape[1]):
        w_gradient[i] = (x[:,i]*(y_pred-y)).mean()
    return w_gradient,b_gradient

def gradient_descent(x,y,w_init,b_init,learning_rate,cost_function,gradient_function,run_iter,p_iter=1000):
    c_hist = []
    w_hist = []
    b_hist = []
    w = w_init
    b = b_init
    for i in range(run_iter):
        w_gradient,b_gradient = gradient_function(x,y,w,b)
        w = w - w_gradient*learning_rate
        b = b - b_gradient*learning_rate
        cost = cost_function(x,y,w,b)

        c_hist.append(cost)
        w_hist.append(w)
        b_hist.append(b)

        if (i%p_iter) == 0:
            print(f'Iteration {i:5}:Cost {cost:.2f},w:{w},b:{b:.2f},w_gradient:{w_gradient}')

    return w,b,w_hist,b_hist,c_hist

w_init = np.array([1,2,2,4])
b_init = 0
learning_rate = 1.0e-2

w_final,b_final,w_hist,b_hist,c_hist = gradient_descent(x_train,y_train,w_init,b_init,learning_rate,computer_cost,computer_gradient,10000)
print('the final w is {0},the final b is {1}'.format(w_final,b_final))

# Validate final w/b
y_pred = (w_final*x_test).sum(axis=1) + b_final
test_pd = pd.DataFrame(
    {
        'y_pred':y_pred,
        'y_test':y_test
    }
)
print(test_pd)
print(computer_cost(x_test,y_test,w_final,b_final))

# predict: 5.3, Master, CityA
x_real = np.array([[5.3,2,1,0]])
x_real = scarler.transform(x_real)
y_real = (w_final*x_real).sum(axis=1) + b_final
print('Prediction for the candidate: 5.3, Master, CityA ')
print(f'The salary for this candidate should be {y_real}K')