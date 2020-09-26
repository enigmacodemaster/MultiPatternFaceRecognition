import numpy as np
import matplotlib.pyplot as plt

# 求x^2的导数
def dJ(x):
    return 2 * x

# 定义一个求函数值的函数J
def J(x):
    try:
        return x ** 2
    except:
        return float('inf')

''' 算法部分 '''
x = -2.0 # 初始值
learning_rate = 0.1 # 学习率
epsilon = 1e-8 # 阈值
history = [] # 记录更新情况
cnt = 0 # 记录迭代步数

while True:
    gradient = dJ(x) # 导数(梯度)
    last_x = x 
    x = x - learning_rate * gradient # 梯度更新
    cnt += 1
    history.append(last_x)
    if (abs(J(last_x) - J(x))) < epsilon: # 终止条件
        break

# print(float('inf') == float('inf'))

plot_x = np.linspace(-3, 3, 200)
plot_y = plot_x ** 2

# print(history)
print('iter %d steps to final result'%(cnt))
print('y = x^2 的最小值为： %d'%(J(x)))
plt.plot(plot_x, plot_y)
plt.plot(np.array(history), J(np.array(history)), color='r', marker='*')
plt.show()