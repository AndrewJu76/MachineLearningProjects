from sklearn.datasets import make_blobs
import torch
import numpy as np
import matplotlib.pyplot as plt


def hypothesis(theta0, theta1, theta2, x1, x2):
    z = theta0 + theta1 * x1 + theta2 * x2
    h = torch.sigmoid(z)
    return h.view(-1, 1)


def J(h, y):
    return -torch.mean(y*torch.log(h) + (1 - y) * torch.log(1 - h))


if __name__ == "__main__":
    #生存样本
    x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)
    x1 = x[:, 0]
    x2 = x[:, 1]

    plt.scatter(x1[y == 1], x2[y == 1], color='blue', marker='o')
    plt.scatter(x1[y == 0], x2[y == 0], color='red', marker='x')
    # plt.show()

    # 特征x1,x2 and 标签y,转为张量
    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # 定义直线的3个参数
    theta0 = torch.tensor(0.0, requires_grad=True)
    theta1 = torch.tensor(0.0, requires_grad=True)
    theta2 = torch.tensor(0.0, requires_grad=True)

    # 定义优化器
    optimizer = torch.optim.Adam([theta0, theta1, theta2])

    for epoch in range(10000): #逻辑回归迭代
        # 计算模型预测值
        h = hypothesis(theta0, theta1, theta2, x1, x2)
        # 计算损失函数
        loss = J(h, y)
        # 计算损失函数关于参数的梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 将梯度清零
        optimizer.zero_grad()

        if epoch % 1000 == 0:
            # 每1000次，打印一次损失值
            print(f'After {epoch} iterations, the loss is {loss.item():.3f} ')


    w1 = theta1.item()
    w2 = theta2.item()
    b = theta0.item()
    x = np.linspace(-1,6,100)
    d = -(w1*x+b)*1.0/w2
    plt.plot(x,d)
    plt.show()