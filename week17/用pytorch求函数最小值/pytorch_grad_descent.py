import torch
from torch import optim
import matplotlib.pyplot as plt

from math import pi

tensor_1 = torch.tensor([5., 3.])
tensor_2 = torch.tensor([-3., -4.])

x = torch.tensor([1.,1.], requires_grad=True)
optimizer = optim.SGD([x,], lr=0.01, momentum=0) # 学习率设置为0.01比较好
for step in range(30):
    if step:
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 损失函数的求导
        optimizer.step() # 计算一次梯度下降更新的过程
    loss = (torch.mul(tensor_1, x).sum() - 1.)** 2 +  (torch.mul(tensor_2, x).sum() + 1.) ** 2
    print('step = {}: x = {}, f(x) = {}'.format(step, x.tolist(), loss))