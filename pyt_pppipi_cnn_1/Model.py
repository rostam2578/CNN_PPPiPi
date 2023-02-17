# Defining the network
print('\n\n\n\n', 'The Network ...', '\n\n')
from DataLoad import *

import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#########################################################
modelname = 'pyt_pppipi_cnn_1'
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 3, 1)
        self.conv2 = nn.Conv2d(256, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 64, 3, 1)
        self.fc1 = nn.Linear(667776, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 6796)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

vars()[modelname] = Net().to(device)
net = vars()[modelname]
print(net)
print('\nnumber of the free parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))

for name, param in net.named_parameters():
    print('parameters of the network', name, '\n', param.data.shape, '\n', param.requires_grad , '\n', param.data, '\n', param)

#########################################################
# one sample event and passing it from the network before training
EvBTr = 1007
result1 = net(TraTen.reshape(-1,1,43,288)[1000:1020].to(device))

fig = plt.figure(figsize=(40, 21))
plt.rcParams['font.size'] = '18'
ax1 = fig.add_subplot(331)
fig.colorbar(ax1.matshow(traincnnpppipi_rectangle[EvBTr], aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBTr} with noise \n color indicates time')

ax2 = fig.add_subplot(332)
fig.colorbar(ax2.matshow(traincnnpppipi_rectangle[EvBTr], aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBTr} without noise \n color indicates time')

ax3 = fig.add_subplot(333)
fig.colorbar(ax3.matshow(sitonsquare(result1[EvBTr - 1000]), aspect=2, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBTr} passed through network before training \n color indicates time')
print(f'\nPassing event {EvBTr} from the network before training', 'input', TraTen[EvBTr], '\nresult1:', result1, '\nresult1.shape:', result1.shape)

#########################################################
#from torch.utils.data import DataLoader
#TraTenBat = DataLoader(TraTen, batch_size=2, shuffle=True)

#########################################################
# Passing a batch from the network before training
BATCH_SIZE = 20
EvBBTr = 7
batten = TraTen.reshape(-1,1,43,288)[0:BATCH_SIZE]
result2 = net(batten.to(device))
print(f'\nPassing two random events from the network before training', '\nresult1:', result1, '\nresult1.shape:', result1.shape, '\ninput:', traincnnpppipi_rectangle[EvBTr])

# the first event
ax4 = fig.add_subplot(334)
fig.colorbar(ax4.matshow(traincnnpppipi_rectangle[EvBBTr], aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBBTr} with noise \n color indicates time')

# without noise
ax5 = fig.add_subplot(335)
fig.colorbar(ax5.matshow(sitonsquare(TrvTen[EvBBTr]), aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBBTr} with noise \n color indicates time')

# pass from the net
ax6 = fig.add_subplot(336)
fig.colorbar(ax6.matshow(sitonsquare(result2[EvBBTr]), aspect=2, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event {EvBBTr} passed through network in a batch before training \n color indicates time')

# the second event
ax7 = fig.add_subplot(337)
fig.colorbar(ax7.matshow(traincnnpppipi_rectangle[EvBBTr + 1], aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBBTr + 1} with noise \n color indicates time')

# without noise
ax8 = fig.add_subplot(338)
fig.colorbar(ax8.matshow(sitonsquare(TrvTen[EvBBTr + 1]), aspect=2, vmin=0, vmax=1, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event number {EvBBTr + 1} with noise \n color indicates time')

# pass from the net
ax9 = fig.add_subplot(339)
fig.colorbar(ax9.matshow(sitonsquare(result2[EvBBTr + 1]), aspect=2, extent=[0, 288, 0, 43], origin='lower')\
                ,fraction=0.014)                
plt.xlabel('cell')
plt.ylabel('layer')
plt.xlim(0, 288)
plt.ylim(0, 43)
plt.title(f'event {EvBBTr + 1} passed through network in a batch before training \n color indicates time')

#########################################################
t = datetime.datetime.now()
plt.savefig(f'/hpcfs/bes/mlgpu/hoseinkk/MLTracking/otherparticles/pppipi/pyt_pppipi_cnn_1/results/{t}\
    passing three random events {EvBTr, EvBBTr, EvBBTr + 1} from network before training.png', bbox_inches='tight')

#########################################################
# message passing example for multi dimentional node feature:
#g = dgl.graph(([0, 0, 0, 1], [1, 2, 3, 2]))
#g.edata['efet'] = torch.tensor([0.7, 0.7, 0.7, 0.7])
#tens1 = torch.tensor(([0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]))
#g.ndata['h'] = torch.transpose(tens1, 0, 1)
#g.update_all(fn.u_mul_e('h', 'efet', 'm'), fn.sum('m', 'h1'))
#g = dgl.add_reverse_edges(g)
#g.edata['efet'] = torch.tensor([0.7] * 8)
#g.update_all(fn.u_mul_e('h', 'efet', 'm'), fn.sum('m', 'h2'))