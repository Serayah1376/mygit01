# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 22:58:51 2021

@author: 10983
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #定义第一个卷积层，输入维度为1，输出维度为6，卷积核大小为3*3
        self.conv1=nn.Conv2d(1,6,3)
        #定义第二个
        self.conv2=nn.Conv2d(6,16,3)
        #定义三层全连接神经网络
        self.fc1=nn.Linear(16*6*6,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        
    def forward(self,x):
        #注意卷积后面要加激活层和池化层
        x=F.max_pool2d(F.relu(self.conv1(x)),kernel_size=2)
        x=F.max_pool2d(F.relu(self.conv2(x)),kernel_size=2)
        #经过卷积层处理后，张量要进入全连接层，进入前需要调整张量的形状
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return x
    
    def num_flat_features(self,x):
        size=x.size()[1:]  #取后面三个参数
        num_features=1
        for s in size:
            num_features*=s
        return num_features

net=Net()

input=torch.randn(1,1,32,32)
out=net(input)

target=torch.randn(10)
target=target.view(1,-1)#变成矩阵或者说张量？？

criterion=nn.MSELoss()
optimizer=optim.SGD(net.parameters(),lr=0.01)
loss=criterion(out,target)
net.zero_grad()
optimizer.zero_grad()
loss.backward()




        
  
        
  
    
  
    
  
    
  
    
  
    
        