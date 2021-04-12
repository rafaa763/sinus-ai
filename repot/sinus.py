from pylab import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import torch.optim as optim

z=3.14-1.57
taille=100
z=z/taille
x = 1.57*torch.ones(taille)
for i in range(taille):
    x[i]=x[i]+z*i
print(x)
trainset = torch.utils.data.DataLoader(x, batch_size=8, shuffle=True)
testset = torch.utils.data.DataLoader(x, batch_size=8, shuffle=True)
print (x)
y=torch.sin(x)
print (y)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8) 
        self.fc2 = nn.Linear(8, 8)   
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)   
        self.fc5 = nn.Linear(8, 1)    
    
    def forward(self, x):
        x = F.sigmoid(self.fc1(x)) 
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x)
        
        return torch.sigmoid(x)
net = Net()
print(net)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(800): 
    for data in trainset: 
        X = data  
        output = net(X.view(-1,1))
        target = torch.sin(X)
        target = target.view(-1,1).float()
        loss = F.mse_loss(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
print(loss)
predictions = net(torch.Tensor(np.reshape(x, (-1,1))))
plt.plot(x, y , x ,predictions.detach().numpy())
plt.show()
