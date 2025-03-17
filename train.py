import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader,random_split
from model import LLMConfig,LLM,MyDataset
model = LLM(LLMConfig)
MyDataset = MyDataset(LLMConfig)
train_set ,val_set = torch.utils.data.random_split(MyDataset, [0.9,0.1])
train_load = DataLoader(train_set,batch_size=3,shuffle=True)
val_load = DataLoader(val_set,batch_size=3,shuffle=True)
optimizer = optim.AdamW(model.parameters(),lr=3e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

def train(model, device, train_loader, optimizer,scheduler, epoch):
    for i in range(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, loss = model(data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def eval(model, device, val_loader):
    model.eval()
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        output, loss =  model(data)
        print(f'LLM的回答是：{output},此时的交叉熵损失是:{loss}')
        
  