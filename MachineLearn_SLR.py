import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

url = ".\\Samples\\Salary_Data.csv"
data = pd.read_csv(url)
x=data['YearsExperience']
y=data['Salary']

# simple linear regression
def computer_gradient(x, y, w, b):
    w_gradient = (x*(w*x+b-y)).mean()
    b_gradient = ((w*x+b-y)).mean()
    return w_gradient,b_gradient

def computer_cost(x,y,w,b):
    y_pred = w*x + b
    cost = (y-y_pred)**2
    cost = cost.sum() / len(x)
    return cost

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
            print(f'Iteration {i:5}:Cost {cost:.2f},w:{w:.2f},b:{b:.2f}')

    return w,b,w_hist,b_hist,c_hist

w_init = -100
b_init = -100
learning_rate = 1.0e-3

w_final,b_final,w_hist,b_hist,c_hist = gradient_descent(x,y,w_init,b_init,learning_rate,computer_cost,computer_gradient,20000)
print('the final w is {0},the the final b is {1}'.format(w_final,b_final))

ws = np.arange(-100,101)
bs = np.arange(-100,101)
costs = np.zeros((201,201))

i = 0
for w in ws:
    j = 0
    for b in bs:
        cost =computer_cost(x,y,w,b)
        costs[i,j] = cost
        j = j + 1
    i = i + 1

ax = plt.axes(projection='3d')
ax.view_init(20,-65)
ax.xaxis.set_pane_color((1,1,1))
ax.yaxis.set_pane_color((1,1,1))
ax.zaxis.set_pane_color((1,1,1))

b_grid,w_grid = np.meshgrid(bs,ws)
ax.plot_surface(w_grid,b_grid,costs,alpha=0.3)
w_index,b_index = np.where(costs == np.min(costs))
ax.scatter(ws[w_index],bs[b_index],costs[w_index,b_index],color='red',s=40)
ax.scatter(w_hist[0],b_hist[0],c_hist[0],color='green',s=40)
ax.plot(w_hist,b_hist,c_hist)
plt.show()

# plt.plot(np.arange(0,100),c_hist[0:100])
# plt.title('iteration vs cost')
# plt.xlabel('iteration')
# plt.ylabel('cost')
# plt.show()

# url = ".\\Samples\\Salary_Data.csv"
# data = pd.read_csv(url)
# # y = w*x+b
# x=data['YearsExperience']
# y=data['Salary']
#
# def plot_pred(w,b):
#     y_pred = w*x + b
#     plt.scatter(x,y_pred,color='blue',label='预测线')
#     plt.scatter(x,y,marker='x',color='red',label='真实数据')
#     plt.title('年资--薪水')
#     plt.ylabel('年资')
#     plt.xlabel('薪水(千)')
#     plt.xlim([0,12])
#     plt.ylim([-60,140])
#     plt.legend()
#     plt.show()
#
# plot_pred(9.2,27.4)